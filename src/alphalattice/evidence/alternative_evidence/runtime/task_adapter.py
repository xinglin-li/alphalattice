"""Task Control adapter for the evidence refresh: eight stages, one extraction.

The stages are the recovery boundaries that matter: acquisition (network),
canonicalization, indexing (embedding) and the single actor execution each
seal an artifact before the next begins, so a restart resumes after the last
expensive step rather than repeating it. Span selection is a Host step with
its own receipt, which is what makes the packet an actor read reproducible.

A coverage run is the same adapter over several units: one Task whose work
items are the same stages qualified by unit (`u01_admit_evidence_request` ...
`u07_select_evidence_spans`), each unit its own request of at most eight
issuers, executed one after another by the one runner. A unit that refuses
seals a typed failure and the run goes on to the next unit; a unit whose
intent was already prepared -- by an earlier run, a cancelled run or a single
preparation -- carries that preparation's sealed artifacts forward instead
of computing them again. Nothing here schedules: the plan is the schedule.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import partial
from typing import Self, cast
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from pydantic import Field, model_validator
from pydantic_core import to_jsonable_python

from alphalattice.control.task_control.contracts import (
    FAILURE_CODE_MAX_LENGTH,
    PLAN_WORK_ITEM_LIMIT,
    ResearchGoal,
    ResearchPlan,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskStageReceipt,
    WorkItemDefinition,
    failure_code_from,
)
from alphalattice.control.task_control.registry import DuckDbTaskControlRegistry
from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
from alphalattice.kernel.knowledge._embeddings import (
    VERIFIED_PACKS,
    before_first_inference,
)
from alphalattice.kernel.knowledge.retrieval_errors import KnowledgeRetrievalError
from alphalattice.kernel.shared_kernel.identity import canonical_hash, schema_structure
from alphalattice.kernel.shared_kernel.identity_successors import is_current
from alphalattice.protocols.actor_execution import ActorKind, ActorSubmissionBinding
from alphalattice.protocols.actor_execution.answers import AnswerProblem

from ..analysis.contracts import (
    AlternativeEvidenceAnalystAnswer,
    AlternativeEvidenceAnalystBrief,
    AlternativeEvidenceAnalystBriefReceipt,
    AlternativeEvidenceAnalystBriefSubmission,
    AlternativeEvidenceResearchObligation,
    AlternativeEvidenceRetrievalAccessReceipt,
)
from ..analysis.cro_package import compile_cro_alternative_evidence_package
from ..analysis.packet import AlternativeEvidencePacket, render_evidence_packet, span_aliases
from ..analysis.submissions import seal_alternative_evidence_analyst_brief
from ..contracts import (
    ADMITTED_DOCUMENT_CAPACITY,
    SOURCE_FAMILY_BY_EVIDENCE_CLASS,
    AlternativeEvidenceAdmission,
    AlternativeEvidenceClass,
    AlternativeEvidenceContract,
    AlternativeEvidenceMode,
    AlternativeEvidenceReadFiling,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    SecIssuerRegistrySnapshot,
    seal_contract,
)
from ..documents.canonicalization import SecTableCarry
from ..documents.contracts import (
    AlternativeEvidenceDocumentReference,
    AlternativeEvidenceDocumentSet,
)
from ..documents.structure import DocumentStructure
from ..documents.tables import TableViewError, render_table_view, table_catalogue
from ..publication.analysis import AlternativeEvidenceAnalysisPublicationView
from ..publication.contracts import ANALYSIS_POLICY_ROLE, BINDING_ROLES, DECISION_POLICY_ROLE
from ..retrieval.contracts import (
    AlternativeEvidenceResolvedSpan,
    AlternativeEvidenceResolvedSpanSet,
    RetrievalGenerationRecord,
    WholeFilingsGeneration,
)
from ..retrieval.session import MATTER_VIEW_BYTES
from ..sources.acquisition import AlternativeEvidenceTaskCancelled
from ..sources.contracts import (
    AcquiredEvidenceSourceReferenceSet,
    AcquiredEvidenceSourceSet,
    SecFilingInventoryRead,
    SecFilingSelectionPlan,
    inventory_read_key,
)
from ..sources.recorded import RecordedEvidenceDocument, RecordedEvidenceSource
from ..sources.sec_edgar import SecEdgarSource
from .coverage import (
    COVERAGE_RUN_PURPOSE,
    UNKNOWN_SELECTION_RESERVATION,
    AlternativeEvidenceCoverageRun,
    AlternativeEvidenceCoverageUnit,
    AlternativeEvidenceUnitFailure,
    LogicalSourceCounts,
    UnitSourcesShort,
    index_quiet,
    seal_unit_failure,
    sources_short,
)
from .execution import PREPARATIONS, machine_load, plan_execution
from .history import require_answer_format
from .policy import AdmittedEvidencePolicy
from .progress import ACQUIRE, BUILD, SELECT, PreparationProgress
from .service import (
    AlternativeEvidenceAnalysisActor,
    AlternativeEvidenceDocumentIntelligenceRuntime,
    SubmittedAlternativeEvidenceAnalysisActor,
)

TASK_KIND = "alternative_evidence.document_intelligence"
INPUT_SCHEMA_ID = "alternative-evidence-document-intelligence-authority"
UNIT_FAILURE_KIND = "alternative_evidence_unit_failure"
COVERAGE_RUN_CATEGORY = "evidence-coverage-runs"
UNIT_FAILURE_CATEGORY = "evidence-unit-failures"
_PREPARATION_STAGES = frozenset(
    (
        "admit_evidence_request",
        "resolve_official_sources",
        "acquire_source_evidence",
        "canonicalize_documents",
        "build_retrieval_generation",
        "select_evidence_spans",
    )
)
_PREPARED_PURPOSES = frozenset(("PREPARE_PACKET", "CONTINUE_READING"))
"""Single Tasks whose success is a prepared packet: an initial preparation,
or a continuation that read more of the same packet's source."""
_STAGES = (
    ("admit_evidence_request", (), "alternative_evidence_request"),
    ("resolve_official_sources", ("admit_evidence_request",), "alternative_evidence_registry"),
    (
        "acquire_source_evidence",
        ("resolve_official_sources",),
        "alternative_evidence_source_document_set",
    ),
    (
        "canonicalize_documents",
        ("acquire_source_evidence",),
        "alternative_evidence_document_set",
    ),
    (
        "build_retrieval_generation",
        ("canonicalize_documents",),
        "alternative_evidence_retrieval_generation",
    ),
    (
        "select_evidence_spans",
        ("build_retrieval_generation",),
        "alternative_evidence_access_receipt",
    ),
    ("analyze_evidence", ("select_evidence_spans",), "alternative_evidence_analyst_brief"),
    (
        "publish_evidence_analysis",
        ("analyze_evidence",),
        "alternative_evidence_analysis_publication",
    ),
)


class SubmittedEvidenceAnalysis(AlternativeEvidenceContract):
    """Host-bound external answer; durable Task input, not model authority."""

    prepared_task_id: UUID
    prepared_unit_id: str | None = Field(default=None, pattern=r"^u[0-9]{2,3}$")
    """The unit of a coverage run whose packet this answers; None for a single
    preparation."""
    packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer: AlternativeEvidenceAnalystAnswer | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """The accepted part of the actor's answer."""
    dropped: tuple[AnswerProblem, ...] = Field(
        default=(), max_length=256, exclude_if=lambda v: v == ()
    )
    """What the Host did not accept of that answer, and why."""
    submission: AlternativeEvidenceAnalystBriefSubmission | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    """A whole brief as an actor wrote it before the answer format: Tasks
    admitted that way read back for their lineage, and one not yet executed
    is refused by name (`alternative_evidence.submission_format_retired`)."""
    actor_submission: ActorSubmissionBinding

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_submission(self) -> Self:
        """Validate the actor and exact identity of the submitted answer."""
        if self.actor_submission.actor_kind not in {ActorKind.HUMAN, ActorKind.EXTERNAL_AUTOMATION}:
            raise ValueError("alternative_evidence.external_submission_entry_required")
        written = self.answer if self.answer is not None else self.submission
        if (
            written is None
            or (self.answer is not None and self.submission is not None)
            or (self.dropped and self.answer is None)
        ):
            raise ValueError("alternative_evidence.external_submission_invalid")
        if self.actor_submission.submission_hash != _hash(written.model_dump(mode="json")):
            raise ValueError("alternative_evidence.external_submission_identity_invalid")
        return self


class EvidenceContinuation(AlternativeEvidenceContract):
    """Describe one bounded source-reading continuation of a prepared packet.

    Durable Task input names the packet by the receipt and
    span set that packet sealed, and the cumulative allowance the chain is
    admitted under. Never a position: the sealed receipt proves what was
    read, and the reader resumes after it.
    """

    prepared_task_id: UUID
    prepared_unit_id: str | None = Field(default=None, pattern=r"^u[0-9]{2,3}$")
    continuation_of: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The access receipt whose matter record this session continues."""
    continuation_spans: str = Field(pattern=r"^[0-9a-f]{64}$")
    """The resolved span set that receipt delivered."""
    session_limit: int = Field(ge=1, le=99)
    window_limit: int = Field(ge=1, le=4096)
    """The chain's cumulative allowance: matter-reading sessions including the
    initial preparation, and matter windows over every session."""


@dataclass(frozen=True, slots=True)
class AlternativeEvidenceDocumentTaskResources:
    """Host resources never serialized into the Task or the model context."""

    recorded_registry: SecIssuerRegistrySnapshot | None = None
    recorded_documents: tuple[RecordedEvidenceDocument, ...] = ()
    live_source: SecEdgarSource | None = None
    analysis_actor: AlternativeEvidenceAnalysisActor | None = None
    minimum_entity_coverage: float = 0.0
    admitted_authority_hash: str | None = None
    recorded_document_bundle_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate coverage and optional resource binding hashes."""
        if not 0.0 <= self.minimum_entity_coverage <= 1.0:
            raise ValueError("alternative_evidence.minimum_entity_coverage_invalid")
        if self.admitted_authority_hash is not None and (
            len(self.admitted_authority_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.admitted_authority_hash)
        ):
            raise ValueError("alternative_evidence.admitted_authority_hash_invalid")
        if self.recorded_document_bundle_hash is not None and (
            len(self.recorded_document_bundle_hash) != 64
            or any(value not in "0123456789abcdef" for value in self.recorded_document_bundle_hash)
        ):
            raise ValueError("alternative_evidence.recorded_document_bundle_hash_invalid")

    @property
    def binding_hash(self) -> str:
        """Return the identity of Host resources supplied to this Task."""
        actor = self.analysis_actor
        return _hash(
            {
                "admitted_authority_hash": self.admitted_authority_hash,
                "recorded_registry_hash": (
                    None if self.recorded_registry is None else self.recorded_registry.registry_hash
                ),
                "recorded_documents": (
                    self.recorded_document_bundle_hash
                    if self.recorded_document_bundle_hash is not None
                    else tuple(value.model_dump(mode="json") for value in self.recorded_documents)
                ),
                "live_source": (
                    None
                    if self.live_source is None
                    else (
                        f"{type(self.live_source).__module__}.{type(self.live_source).__qualname__}"
                    )
                ),
                "analysis_process_binding_hash": (
                    None if actor is None else actor.process_binding_hash
                ),
                "minimum_entity_coverage": self.minimum_entity_coverage,
            }
        )

    @property
    def preparation_binding_hash(self) -> str:
        """Source work has no dependency on an optional managed analyst process."""
        return replace(self, analysis_actor=None).binding_hash


def alternative_evidence_document_task_contract(
    *,
    request: AlternativeEvidenceRequest,
    admission: AlternativeEvidenceAdmission,
    obligation: AlternativeEvidenceResearchObligation,
    resource_binding_hash: str,
    prepare_only: bool = False,
    submitted_analysis: SubmittedEvidenceAnalysis | None = None,
    continuation: EvidenceContinuation | None = None,
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """Build Task input, goal, and plan for one request.

    The obligation is part of the identity: two Tasks that answered two
    questions are two Tasks with distinct meanings.

    A continuation is one Task of one stage: it reads more source for a
    prepared packet under the packet's own request, obligation and admission,
    and takes the packet's first five stages as its lineage.
    """
    validate_scoped_obligation(request=request, obligation=obligation)
    if sum(1 for flag in (prepare_only, submitted_analysis, continuation) if flag) > 1:
        raise ValueError("alternative_evidence.task_purpose_conflict")
    extra: dict[str, object] = {}
    stages: tuple[tuple[str, tuple[str, ...], str], ...] = _STAGES
    if prepare_only:
        extra = {"purpose": "PREPARE_PACKET"}
        stages = _STAGES[:6]
    elif submitted_analysis is not None:
        extra = {
            "purpose": "SUBMITTED_ANALYSIS",
            "submitted_analysis": submitted_analysis.model_dump(mode="json"),
        }
        stages = ((_STAGES[6][0], (), _STAGES[6][2]), _STAGES[7])
    elif continuation is not None:
        extra = {
            "purpose": "CONTINUE_READING",
            "continuation": continuation.model_dump(mode="json"),
        }
        stages = ((_STAGES[5][0], (), _STAGES[5][2]),)
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id=INPUT_SCHEMA_ID,
        payload={
            "request": request.model_dump(mode="json"),
            "admission": admission.model_dump(mode="json"),
            "obligation": obligation.model_dump(mode="json"),
            "resource_binding_hash": resource_binding_hash,
            **extra,
        },
    )
    reading = prepare_only or continuation is not None
    goal = ResearchGoal.create(
        goal_kind="CONTINUE_ALTERNATIVE_EVIDENCE_READING"
        if continuation is not None
        else "PREPARE_ALTERNATIVE_EVIDENCE_PACKET"
        if prepare_only
        else "PUBLISH_CRO_READY_ALTERNATIVE_EVIDENCE",
        input_hash=envelope.input_hash,
        deliverable_kind="AlternativeEvidenceRetrievalAccessReceipt"
        if reading
        else "AlternativeEvidenceAnalysisPublication",
        summary="Continue bounded source reading of a prepared packet."
        if continuation is not None
        else "Prepare exact evidence for external analysis."
        if prepare_only
        else "Publish governed actor-neutral official-source document intelligence.",
        attributes={
            "mode": request.mode,
            "entity_count": len(request.ordered_entity_ids),
            "portfolio_effect": "NONE",
        },
    )
    work_items = tuple(
        WorkItemDefinition.create(
            stage_id=stage,
            dependency_ids=dependencies,
            verifier_id=f"alternative-evidence.document-intelligence.{stage}",
            required_evidence_kinds=(kind,),
        )
        for stage, dependencies, kind in stages
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=_hash(
            {"owner": __name__, "stages": tuple(value[0] for value in stages)}
        ),
        verifier_catalog_hash=_hash(tuple(value.verifier_id for value in work_items)),
        work_items=work_items,
    )
    return envelope, goal, plan


def coverage_run_task_contract(
    *,
    run: AlternativeEvidenceCoverageRun,
    prepare_only: bool,
) -> tuple[TaskInputEnvelope, ResearchGoal, ResearchPlan]:
    """Build Task input for one coverage run with units as stages.

    The payload names the run (a sealed artifact the adapter reads by hash)
    rather than repeating every unit's request; the work items are the same
    preparation stages, one set per unit in the run's order, and declare no
    required evidence kind because a unit may seal a typed failure in place
    of a stage's artifact -- the adapter's verifier holds that rule.
    """
    stages: tuple[tuple[str, tuple[str, ...], str], ...] = _STAGES[:6] if prepare_only else _STAGES
    if len(run.units) * len(stages) > PLAN_WORK_ITEM_LIMIT:
        # Unreachable while the two bounds agree (`test_a_run_of_many_units
        # _carries_a_plan_task_control_admits`); named here so a run the
        # plan cannot carry is refused by its size, never by a validation
        # error at submission.
        raise ValueError(
            "alternative_evidence.coverage_run_exceeds_plan: "
            f"{len(run.units)} units x {len(stages)} stages > {PLAN_WORK_ITEM_LIMIT}"
        )
    envelope = TaskInputEnvelope.create(
        task_kind=TASK_KIND,
        input_schema_id=INPUT_SCHEMA_ID,
        payload={
            "purpose": COVERAGE_RUN_PURPOSE,
            "run_hash": run.run_hash,
            "scope_hash": run.scope_hash,
            "evidence_as_of": run.evidence_as_of.isoformat(),
            "resource_binding_hash": run.resource_binding_hash,
            "unit_count": len(run.units),
            "entity_count": len(run.ordered_entity_ids),
            "prepare_only": prepare_only,
        },
    )
    goal = ResearchGoal.create(
        goal_kind="PREPARE_ALTERNATIVE_EVIDENCE_COVERAGE"
        if prepare_only
        else "PUBLISH_CRO_READY_ALTERNATIVE_EVIDENCE_COVERAGE",
        input_hash=envelope.input_hash,
        deliverable_kind="AlternativeEvidenceCoverageRun",
        summary=(
            "Prepare exact evidence for every unit of the book."
            if prepare_only
            else "Publish governed document intelligence for every unit of the book."
        ),
        attributes={
            "mode": run.units[0].request.mode,
            "entity_count": len(run.ordered_entity_ids),
            "unit_count": len(run.units),
            "portfolio_effect": "NONE",
        },
    )
    work_items = tuple(
        WorkItemDefinition.create(
            stage_id=f"{unit.unit_id}_{stage}",
            dependency_ids=tuple(f"{unit.unit_id}_{value}" for value in dependencies),
            verifier_id=f"alternative-evidence.document-intelligence.{stage}",
        )
        for unit in run.units
        for stage, dependencies, _kind in stages
    )
    plan = ResearchPlan.create(
        goal_hash=goal.goal_hash,
        workflow_definition_hash=_hash(
            {
                "owner": __name__,
                "purpose": COVERAGE_RUN_PURPOSE,
                "stages": tuple(value[0] for value in stages),
                "units": tuple(unit.unit_id for unit in run.units),
            }
        ),
        verifier_catalog_hash=_hash(
            tuple(dict.fromkeys(value.verifier_id for value in work_items))
        ),
        work_items=work_items,
    )
    return envelope, goal, plan


def _verify_table_view(
    span: AlternativeEvidenceResolvedSpan,
    *,
    text: str,
    reference: AlternativeEvidenceDocumentReference,
    original: tuple[bytes, str, str] | None,
) -> None:
    """Prove a delivered table view again, as the session proved it when
    it was read: the span's range is the placeholder the binding names in
    the canonical text, the retained original behind the document is the
    one the binding names by content hash, and the page rendered from it
    under the same rules and ceiling is the excerpt, byte for byte. A
    view that cannot be proved refuses the packet by the view's own name;
    nothing is re-read from a source and nothing is interpreted.
    """
    binding = span.table_view
    assert binding is not None
    structure = DocumentStructure(text, document_type=reference.document_type)
    placeholder = next(
        (
            value
            for value in table_catalogue(text, structure)
            if value.character_start == span.character_start
            and value.character_end == span.character_end
        ),
        None,
    )
    if placeholder is None or placeholder.ordinal != binding.table_ordinal:
        raise ValueError("alternative_evidence.brief_table_view_placeholder_mismatch")
    if original is None:
        raise ValueError("alternative_evidence.brief_table_view_original_unavailable")
    content, content_sha256, media_type = original
    if (
        content_sha256 != binding.parent_source_content_hash
        or media_type != binding.parent_media_type
        or hashlib.sha256(content).hexdigest() != content_sha256
    ):
        raise ValueError("alternative_evidence.brief_table_view_original_mismatch")
    try:
        view = render_table_view(
            content,
            parent_source_content_hash=content_sha256,
            policy=SecTableCarry(reference.document_type)
            if reference.source_name == "SEC_EDGAR"
            else None,
            placeholder=placeholder,
            text=text,
            rows_from=binding.rows_from,
            byte_ceiling=MATTER_VIEW_BYTES,
        )
    except TableViewError as error:
        raise ValueError(f"alternative_evidence.brief_table_view_unrenderable:{error}") from error
    if (
        view.text != span.excerpt
        or view.rules_id != binding.rules_id
        or view.parser_rules_id != binding.parser_rules_id
        or view.rows_total != binding.rows_total
        or view.rows_to != binding.rows_to
    ):
        raise ValueError("alternative_evidence.brief_span_source_mismatch")


_MODEL_STAGES = frozenset({BUILD, SELECT})
"""The stages whose time is the model's: a later unit's wait for the first."""


def split_unit_stage(stage_id: str) -> tuple[str | None, str]:
    """Split a unit-prefixed stage ID into unit and stage.

    For example, ``u07_select_evidence_spans`` yields ``u07`` and
    ``select_evidence_spans``; a bare stage yields ``None`` and the stage.
    """
    head, _, rest = stage_id.partition("_")
    if len(head) >= 3 and head[0] == "u" and head[1:].isdigit() and rest:
        return head, rest
    return None, stage_id


@dataclass(frozen=True, slots=True)
class UnitAuthority:
    """What one stage may act on: a unit's request, admission and question."""

    unit_id: str | None
    request: AlternativeEvidenceRequest
    admission: AlternativeEvidenceAdmission
    obligation: AlternativeEvidenceResearchObligation


class AlternativeEvidenceDocumentTaskAdapter:
    """Execute and verify document-intelligence Tasks against sealed evidence."""

    task_kind = TASK_KIND

    def __init__(
        self,
        *,
        runtime: AlternativeEvidenceDocumentIntelligenceRuntime,
        registry: DuckDbTaskControlRegistry,
        resources: AlternativeEvidenceDocumentTaskResources,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bind the runtime, Task registry, Host resources, and clock."""
        self.runtime = runtime
        self.registry = registry
        self.resources = resources
        self.clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        """The service's clock, against which a live acquisition's deadline is
        judged while it runs -- the same clock that stamps the Task."""
        self._runs: dict[str, AlternativeEvidenceCoverageRun] = {}
        self._carried_runs: dict[str, list[AlternativeEvidenceCoverageRun]] | None = None
        """Coverage runs by hash: sealed, content-addressed, read once."""
        self._task_registry: tuple[UUID, SecIssuerRegistrySnapshot] | None = None
        """The official registry the current live Task captured, for its later
        units; one slot, so a long-lived service holds one registry, not one
        per Task it ever ran."""
        self._inventory_registry: tuple[datetime, SecIssuerRegistrySnapshot] | None = None
        """The registry the last inventory read its filing indexes with, by its
        cutoff: the preparation's Task at that cutoff takes the same capture, so
        the plans it reuses name the issuers the registry it reads names."""
        self._completed_units: dict[str, tuple[UUID, str | None]] = {}
        """Preparation intent -> where it was completed; only completions are kept."""
        self._absent_units: set[tuple[str, UUID]] = set()
        """(intent, task) pairs whose intent was completed nowhere else when that
        Task asked: a Task's own later stages need not ask again, and only that
        Task's own receipts could change the answer for it."""
        self.progress = PreparationProgress(clock=self.clock)
        """How far each running stage is, for the reads that ask mid-run and the
        Host's progress publisher: telemetry, never a unit's state."""
        self._receipts: dict[UUID, tuple[int, tuple[TaskStageReceipt, ...]]] = {}
        """Stage receipts by task, valid for one task version: a receipt is
        committed in the same transaction that moves the version, so the same
        version is the same receipts, however many readers ask."""

    def _stage_receipts(self, task: TaskRecord) -> tuple[TaskStageReceipt, ...]:
        """This Task's committed stage receipts, read once per task version."""
        cached = self._receipts.get(task.task_id)
        if cached is not None and cached[0] == task.version:
            return cached[1]
        receipts = tuple(self.registry.stage_receipts(task.task_id))
        self._receipts[task.task_id] = (task.version, receipts)
        return receipts

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """Report the Task's execution policy and runtime compatibility."""
        if self._is_run(task):
            run = self._run(task)
            policy: dict[str, object] = {
                "run_hash": run.run_hash,
                "admission": {
                    "network_consent": run.network_consent,
                    "admit_live_official": run.admit_live_official,
                    "admit_model_review": run.admit_model_review,
                },
                "analysis_policy": self.runtime.analysis_policy_hash,
            }
        else:
            request, admission = self._authority(task)
            policy = {
                "request": request.model_dump(mode="json"),
                "admission": admission.model_dump(mode="json"),
                "analysis_policy": self.runtime.analysis_policy_hash,
            }
        return TaskExecutionCompatibility.create(
            task_contract_hash=_hash(
                {
                    "input": schema_structure(TaskInputEnvelope),
                    "goal": schema_structure(ResearchGoal),
                    "plan": schema_structure(ResearchPlan),
                }
            ),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=_hash(policy),
            framework_identity_hash=_hash({"task_control": "document-intelligence-eight-stage"}),
        )

    def concurrent_work_items(self, task: TaskRecord) -> int:
        """Return the allowed concurrent work items for a Task.

        A coverage run's units are independent. As many prepare at once as the
        CPU budget allows, with each model session using its share of cores
        (`runtime.execution`: the first unit alone on more, then later units
        at the safe width).
        Every session proves the retrieval canary when it loads, so no number
        moves. The plan and why are recorded with the machine as found. One
        request is one at a time.
        """
        if not self._is_run(task):
            return 1
        plan = plan_execution(
            self.runtime.execution.read(),
            machine_load(preparations_running=self.preparations_running()),
            units=len(self._run(task).units),
            task_id=task.task_id,
            planned_at=datetime.now(UTC),
        )
        self.runtime.execution.record(plan)
        PREPARATIONS.begin(task.task_id, plan)
        return plan.units_at_once

    def preparations_running(self) -> int:
        """Count book preparations currently running in this Host."""

        def alive(task_id: UUID) -> bool:
            try:
                return self.registry.task(task_id).lifecycle is TaskLifecycle.RUNNING
            except (KeyError, ValueError):
                return False

        return PREPARATIONS.running(alive)

    def execute_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
    ) -> StageExecutionResult:
        """Execute one planned Task stage and return its evidence."""
        del execution
        rank = self._rank(task, work_item.stage_id)
        stage = split_unit_stage(work_item.stage_id)[1]
        later = rank > 0 and stage in _MODEL_STAGES
        if later and stage == SELECT:
            self._after_the_first_unit(task)
        # A later unit's build waits only at its first model call (F2): its reading,
        # cutting and session loading run while the first unit prepares. Its sessions
        # are its own share of the CPU budget, the first unit's its whole plan.
        gate = partial(self._after_the_first_unit, task) if later and stage == BUILD else None
        # One stage at a time holds the runtime's shared state; a stage gives it
        # up only where it touches none (`EvidenceWriter`).
        with (
            VERIFIED_PACKS.threads_for(PREPARATIONS.threads(task.task_id, first=rank == 0)),
            before_first_inference(gate),
            self.runtime.writer.held(rank=rank),
        ):
            return self._execute_stage(task, work_item)

    def _after_the_first_unit(self, task: TaskRecord) -> None:
        """A later unit's model work waits until the book's first unit is
        prepared or has failed: the first unit reads as fast as it would alone
        -- its bundle is the one the lead starts on -- while the later units
        acquire, canonicalize and reach their first model call meanwhile, and
        then share the cores. Measured on the median book with every unit's
        model stages at once, the first unit was prepared at 162 s against
        107 s one at a time.
        """
        first = self._run(task).units[0].unit_id
        while True:
            current = self.registry.task(task.task_id)
            if current.lifecycle is not TaskLifecycle.RUNNING:
                return
            if self.unit_states(current)[first]["state"] != "PENDING":
                # The later units' sessions take their share of the budget.
                PREPARATIONS.open(task.task_id)
                return
            time.sleep(0.5)

    def _rank(self, task: TaskRecord, stage_id: str) -> int:
        """A unit's place in its run (heaviest first): who goes first at the writer."""
        unit_id, _stage = split_unit_stage(stage_id)
        if unit_id is None or not self._is_run(task):
            return 0
        ids = [unit.unit_id for unit in self._run(task).units]
        return ids.index(unit_id) if unit_id in ids else len(ids)

    def _execute_stage(
        self, task: TaskRecord, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        unit_id, stage = split_unit_stage(work_item.stage_id)
        authority = self.unit_authority(task, unit_id)
        request, admission = authority.request, authority.admission
        now = task.updated_at
        if unit_id is not None:
            # A unit that already refused carries its refusal through the rest
            # of its stages; a unit whose intent was prepared elsewhere carries
            # that preparation's artifacts instead of computing them again.
            failed = self._unit_failure_evidence(task, unit_id)
            if failed is not None:
                return StageExecutionResult(disposition=StageDisposition.READY, evidence=(failed,))
            source = self._completed_unit_source(task, authority)
            if source is not None:
                copied = self._copied_evidence(source, stage)
                if copied is not None:
                    return StageExecutionResult(
                        disposition=StageDisposition.READY, evidence=(copied,)
                    )
        count = self.progress.begin(
            task.task_id, unit_id, stage, total=self._planned(task, unit_id, stage)
        )
        try:
            if stage == "admit_evidence_request":
                self.runtime.acquisition.admit(request=request, admission=admission, now=now)
                evidence = _evidence("alternative_evidence_request", request.request_hash)
            elif stage == "resolve_official_sources":
                registry = self._resolve_registry(
                    request=request, observed_at=now, task_id=task.task_id
                )
                evidence = _evidence("alternative_evidence_registry", registry.registry_hash)
            elif stage == "acquire_source_evidence":
                snapshot, source_set = self._acquire(
                    task,
                    request=request,
                    admission=admission,
                    unit_id=unit_id,
                    fetched=count.advance,
                )
                count.finish(len(source_set.documents))
                # The refusal names the coverage reached and needed, and the issuers
                # without a source, which the unit's failure keeps (V541); an issuer whose
                # index at the cutoff showed nothing filed is left out of the share, as the
                # packing and the installer leave it out (V587).
                short = sources_short(
                    request.ordered_entity_ids,
                    {value.entity_id for value in snapshot.citations},
                    floor=self.resources.minimum_entity_coverage,
                    quiet=lambda _uncovered: index_quiet(
                        self._sealed_selection_plans(),
                        evidence_as_of=request.evidence_as_of,
                        window_days=request.source_policy.sec_recent_8k_days,
                    ),
                )
                if short is not None:
                    raise short
                evidence = _evidence(
                    "alternative_evidence_source_document_set",
                    source_set.source_set_hash,
                    snapshot=snapshot.snapshot_hash,
                )
            elif stage == "canonicalize_documents":
                source_set = self._source_set(task, unit_id)
                # Keep Workspace revisions byte-identical if Task Control resumes
                # after the immutable revision was written but before the stage
                # receipt was committed.
                count.expect(len(source_set.documents))
                document_set = self.runtime.canonicalize(
                    source_set=source_set, published_at=source_set.acquired_at, began=count.advance
                )
                count.finish(len(document_set.documents))
                evidence = _evidence(
                    "alternative_evidence_document_set", document_set.document_set_hash
                )
            elif stage == "build_retrieval_generation":
                document_set = self._document_set(task, unit_id)
                continuing = task.input.payload.get("purpose") == "CONTINUE_READING"
                generation: RetrievalGenerationRecord
                if not continuing and self.runtime.whole_delivery(request, document_set):
                    # Short enough for the bundle: delivered whole, no index (W4).
                    generation = self.runtime.seal_whole_generation(
                        document_set=document_set, built_at=document_set.published_at
                    )
                else:
                    generation = self.runtime.build_retrieval(
                        document_set=document_set,
                        # The rebuildable generation identity must not depend on the
                        # wall clock of a recovery attempt.
                        built_at=document_set.published_at,
                        # A selection this request reuses whole from a sealed
                        # receipt opens no session: its index is not built here.
                        reuse_for=None if continuing else request,
                        chunks=count,
                    )
                count.finish(generation.chunk_count)
                evidence = _evidence(
                    "alternative_evidence_retrieval_generation", generation.generation_hash
                )
            elif stage == "select_evidence_spans":
                if task.input.payload.get("purpose") == "CONTINUE_READING":
                    continuation = self._continuation(task)
                    prior_receipt, prior_spans = self.continuation_prior(continuation)
                    receipt, spans = self.runtime.continue_evidence(
                        request=request,
                        document_set=self._document_set(task, unit_id),
                        generation=self._generation(task, unit_id),
                        prior_receipt=prior_receipt,
                        prior_spans=prior_spans,
                        session_limit=continuation.session_limit,
                        window_limit=continuation.window_limit,
                    )
                elif isinstance(held := self._generation(task, unit_id), WholeFilingsGeneration):
                    receipt, spans = self.runtime.select_whole(
                        request=request,
                        document_set=self._document_set(task, unit_id),
                        generation=held,
                    )
                else:
                    receipt, spans = self.runtime.select_evidence(
                        request=request,
                        document_set=self._document_set(task, unit_id),
                        generation=held,
                    )
                # Name the span set by its own sealed identity, never by a
                # re-derived one: the store is addressed by the contract's hash.
                span_set = seal_contract(
                    AlternativeEvidenceResolvedSpanSet,
                    "span_set_hash",
                    request_hash=request.request_hash,
                    retrieval_generation_hash=receipt.retrieval_generation_hash,
                    spans=spans,
                )
                evidence = _evidence(
                    "alternative_evidence_access_receipt",
                    receipt.receipt_hash,
                    spans=span_set.span_set_hash,
                )
                count.finish(1)
            elif stage == "analyze_evidence":
                receipt, spans = self._packet_inputs(task, unit_id)
                actor = self.resources.analysis_actor
                if task.input.payload.get("purpose") == "SUBMITTED_ANALYSIS":
                    submitted = self._submission(task)
                    answer = self.validate_submission(submitted, now=now)
                    actor = SubmittedAlternativeEvidenceAnalysisActor(
                        answer=answer,
                        dropped=submitted.dropped,
                        actor_kind=submitted.actor_submission.actor_kind,
                        actor_id=submitted.actor_submission.actor_id,
                    )
                if actor is None:
                    raise ValueError("alternative_evidence.analysis_actor_missing")
                result = self.runtime.analyze(
                    actor=actor,
                    request=request,
                    obligation=authority.obligation,
                    snapshot=self._snapshot(task, unit_id),
                    document_set=self._document_set(task, unit_id),
                    generation=self._generation(task, unit_id),
                    access_receipt=receipt,
                    resolved_spans=spans,
                    completed_at=now,
                )
                evidence = _evidence(
                    "alternative_evidence_analyst_brief",
                    result.brief.brief_hash,
                    decision=result.analyst_receipt.receipt_hash,
                )
            elif stage == "publish_evidence_analysis":
                receipt, spans = self._packet_inputs(task, unit_id)
                brief_evidence = self._stage_evidence(task, "analyze_evidence", unit_id)
                brief = self.runtime.artifacts.load(
                    "analyst-briefs", brief_evidence.content_hash, AlternativeEvidenceAnalystBrief
                )
                analyst_receipt = self.runtime.artifacts.load(
                    "analyst-brief-receipts",
                    _parameter(brief_evidence.reference, "decision"),
                    AlternativeEvidenceAnalystBriefReceipt,
                )
                if brief.obligation_hash != authority.obligation.obligation_hash:
                    raise ValueError("alternative_evidence.task_obligation_not_bound")
                package = compile_cro_alternative_evidence_package(
                    request=request,
                    snapshot=self._snapshot(task, unit_id),
                    document_set=self._document_set(task, unit_id),
                    generation=self._generation(task, unit_id),
                    access_receipt=receipt,
                    brief=brief,
                    resolved_spans=spans,
                    dropped=analyst_receipt.dropped,
                )
                publication = self.runtime.publications.publish(
                    request=request,
                    registry=self._registry(task, unit_id),
                    snapshot=self._snapshot(task, unit_id),
                    document_set=self._document_set(task, unit_id),
                    generation=self._generation(task, unit_id),
                    access_receipt=receipt,
                    analyst_receipt=analyst_receipt,
                    cro_package=package,
                    published_at=now,
                )
                self.runtime.sealed_readings.add(
                    publication.publication,
                    request=request,
                    documents=self._document_set(task, unit_id),
                    brief=brief,
                )
                evidence = _evidence(
                    "alternative_evidence_analysis_publication",
                    publication.publication.publication_hash,
                )
            else:
                raise ValueError("alternative_evidence.document_stage_unknown")
        except AlternativeEvidenceTaskCancelled:
            return StageExecutionResult(
                disposition=StageDisposition.CANCELLED,
                failure_code="alternative_evidence.current_task_cancelled",
            )
        except Exception as error:
            self.progress.drop(task.task_id, unit_id)
            code = _failure_code(error)
            if unit_id is None or _stops_the_run(code):
                return StageExecutionResult(disposition=StageDisposition.BLOCKED, failure_code=code)
            # One unit's refusal is that unit's, sealed by the owner's own code
            # under the stage that refused; the run goes on to the next unit.
            run = self._run(task)
            failure = seal_unit_failure(
                run_hash=run.run_hash,
                unit_id=unit_id,
                ordered_entity_ids=request.ordered_entity_ids,
                stage_id=stage,
                failure_code=_failure_code(error),
                recorded_at=now,
                uncovered_entity_ids=error.uncovered if isinstance(error, UnitSourcesShort) else (),
            )
            self.runtime.artifacts.publish(UNIT_FAILURE_CATEGORY, failure.failure_hash, failure)
            return StageExecutionResult(
                disposition=StageDisposition.READY,
                evidence=(_evidence(UNIT_FAILURE_KIND, failure.failure_hash, unit=unit_id),),
            )
        return StageExecutionResult(disposition=StageDisposition.READY, evidence=(evidence,))

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """Verify a stage's sealed evidence against its Task contract."""
        del execution
        if len(evidence) != 1:
            raise ValueError("alternative_evidence.stage_evidence_invalid")
        item = evidence[0]
        unit_id, stage = split_unit_stage(work_item.stage_id)
        expected_kind = next(kind for name, _deps, kind in _STAGES if name == stage)
        if unit_id is not None and item.evidence_kind == UNIT_FAILURE_KIND:
            failure = self.runtime.artifacts.load(
                UNIT_FAILURE_CATEGORY, item.content_hash, AlternativeEvidenceUnitFailure
            )
            if failure.unit_id != unit_id or failure.run_hash != self._run(task).run_hash:
                raise ValueError("alternative_evidence.unit_failure_invalid")
            return evidence
        if item.evidence_kind != expected_kind:
            raise ValueError("alternative_evidence.stage_evidence_kind_invalid")
        loaders: dict[str, tuple[str, type]] = {
            "resolve_official_sources": ("registries", SecIssuerRegistrySnapshot),
            "canonicalize_documents": ("document-sets", AlternativeEvidenceDocumentSet),
            "select_evidence_spans": (
                "retrieval-access-receipts",
                AlternativeEvidenceRetrievalAccessReceipt,
            ),
            "analyze_evidence": ("analyst-briefs", AlternativeEvidenceAnalystBrief),
        }
        if stage == "admit_evidence_request":
            authority = self.unit_authority(task, unit_id)
            self.runtime.acquisition.admit(
                request=authority.request, admission=authority.admission, now=task.updated_at
            )
        elif stage == "publish_evidence_analysis":
            publication = self.runtime.publications.read(item.content_hash, now=task.updated_at)
            if publication.publication.publication_hash != item.content_hash:
                raise ValueError("alternative_evidence.analysis_publication_invalid")
        elif stage == "build_retrieval_generation":
            self.runtime.artifacts.load_retrieval_generation(item.content_hash)
        elif stage == "acquire_source_evidence":
            self.runtime.artifacts.load_source_set(item.content_hash)
        else:
            category, model = loaders[stage]
            self.runtime.artifacts.load(category, item.content_hash, model)
        return evidence

    # ----------------------------------------------------------- stage helpers

    def published_analysis(
        self, task_id: UUID, *, now: datetime, unit_id: str | None = None
    ) -> AlternativeEvidenceAnalysisPublicationView:
        """Read a completed Task's verified analysis publication."""
        task = self.registry.task(task_id)
        self.unit_authority(task, unit_id)
        if unit_id is None and task.lifecycle is not TaskLifecycle.SUCCEEDED:
            raise ValueError("alternative_evidence.analysis_not_complete")
        return self.runtime.publications.read(
            self._stage_identity(task, "publish_evidence_analysis", unit_id), now=now
        )

    # ---------------------------------------------------------- coverage runs

    def _is_run(self, task: TaskRecord) -> bool:
        return bool(task.input.payload.get("purpose") == COVERAGE_RUN_PURPOSE)

    def _run(self, task: TaskRecord) -> AlternativeEvidenceCoverageRun:
        """The sealed run a coverage Task executes, by the hash its input names."""
        if not self._is_run(task):
            raise ValueError("alternative_evidence.task_not_a_coverage_run")
        run_hash = str(task.input.payload["run_hash"])
        run = self._runs.get(run_hash)
        if run is None:
            run = self.runtime.artifacts.load(
                COVERAGE_RUN_CATEGORY, run_hash, AlternativeEvidenceCoverageRun
            )
            if run.resource_binding_hash != task.input.payload.get("resource_binding_hash"):
                raise ValueError("alternative_evidence.coverage_run_binding_invalid")
            self._runs[run_hash] = run
        return run

    def logical_source_counts(
        self,
        *,
        entities: tuple[str, ...],
        evidence_as_of: datetime,
        mode: AlternativeEvidenceMode,
        evidence_classes: tuple[AlternativeEvidenceClass, ...],
        policy_budget: int,
        window_days: int,
    ) -> LogicalSourceCounts:
        """Count each issuer's logical sources at the requested cutoff.

        Each count reflects the issuer's selection under the policy budget,
        decided before units are packed. Return its basis and issuers with no
        filings in the window. In recorded mode, count the library's cutoff-valid
        documents; a library says nothing of what was filed, so no issuer is
        named as filing nothing. In live mode the issuer's filing index read
        at this cutoff (`read_inventory`) when there is one: exact, and an
        empty plan is an issuer that filed nothing. What a current analysis at
        or before the cutoff read in the window is not counted: it is named as
        read earlier, and an issuer with nothing else in the window is carried
        (W3). Else its latest sealed selection plan at or before the cutoff, or
        the policy budget named as the reservation of an issuer never read --
        no request is made here.
        """
        counts: dict[str, int] = {}
        basis: dict[str, str] = {}
        if mode is AlternativeEvidenceMode.RECORDED:
            classes = frozenset(evidence_classes)
            for entity in entities:
                selected, deferred = RecordedEvidenceSource.logical_selection(
                    entity_id=entity,
                    documents=self.resources.recorded_documents,
                    evidence_as_of=evidence_as_of,
                    evidence_classes=classes,
                    policy_budget=policy_budget,
                )
                counts[entity] = len(selected)
                basis[entity] = (
                    f"recorded library: {len(selected)} cutoff-valid document(s) selected"
                    + (f", {len(deferred)} beyond the policy budget" if deferred else "")
                )
            return LogicalSourceCounts(counts=counts, basis=basis)
        nothing: set[str] = set()
        carried: set[str] = set()
        read_filings: list[AlternativeEvidenceReadFiling] = []
        plans: dict[str, list[SecFilingSelectionPlan]] | None = None
        for entity in entities:
            read = self._inventory_read(
                entity,
                evidence_as_of=evidence_as_of,
                window_days=window_days,
                policy_budget=policy_budget,
            )
            if read is not None:
                window = (
                    *(value.accession for value in read.plan.events),
                    *(
                        value.accession
                        for value in read.plan.deferred
                        if value.reason in {"BEYOND_CAPACITY", "READ_EARLIER"}
                    ),
                )
                readings = self.runtime.sealed_readings.read_by(
                    entity, evidence_as_of=evidence_as_of, window_days=window_days
                )
                earlier = sorted(accession for accession in window if accession in readings)
                read_filings.extend(
                    AlternativeEvidenceReadFiling(
                        entity_id=entity, accession=accession, publication_hash=readings[accession]
                    )
                    for accession in earlier
                )
                unread = len(window) - len(earlier)
                capacity = min(policy_budget, ADMITTED_DOCUMENT_CAPACITY)
                counts[entity] = min(capacity, unread)
                if read.nothing_filed:
                    nothing.add(entity)
                    basis[entity] = (
                        f"filing index read at the cutoff: nothing filed in the last "
                        f"{window_days} days"
                    )
                elif not unread:
                    carried.add(entity)
                    basis[entity] = (
                        f"filing index read at the cutoff: {len(window)} filing(s) in the last "
                        f"{window_days} days, every one read earlier; no new filing"
                    )
                else:
                    beyond = max(0, unread - capacity)
                    basis[entity] = (
                        f"filing index read at the cutoff: {unread} new filing(s) "
                        f"in the last {window_days} days"
                        + (f", {len(earlier)} read earlier" if earlier else "")
                        + (f", {beyond} beyond the capacity" if beyond else "")
                    )
                continue
            if plans is None:
                plans = self._sealed_selection_plans()
            held = [
                plan
                for plan in plans.get(entity, ())
                if plan.evidence_as_of <= evidence_as_of and not plan.accession_scope
            ]
            if not held:
                # Nothing is known of the issuer's inventory yet: the unit
                # reserves the baseline share for it, the acquisition plans
                # under the issuer's own budget, defers what the unit cannot
                # take by name, and the next run packs from the sealed plan.
                counts[entity] = min(policy_budget, UNKNOWN_SELECTION_RESERVATION)
                basis[entity] = (
                    "no sealed selection plan at or before the cutoff: reserved at the "
                    f"baseline share of {UNKNOWN_SELECTION_RESERVATION} until the official "
                    "inventory is read; events the unit cannot take are deferred by name"
                )
                continue
            plan = max(held, key=lambda value: (value.evidence_as_of, value.plan_hash))
            beyond = sum(
                1
                for value in plan.deferred
                if value.reason in {"BEYOND_CAPACITY", "BEYOND_UNIT_CAPACITY"}
            )
            counts[entity] = min(
                policy_budget,
                ADMITTED_DOCUMENT_CAPACITY,
                len(plan.required) + len(plan.events) + beyond,
            )
            basis[entity] = (
                f"sealed selection plan {plan.plan_hash[:12]} at "
                f"{plan.evidence_as_of.isoformat()}: {len(plan.required)} baseline(s), "
                f"{len(plan.events)} selected event(s), {beyond} deferred beyond a capacity"
            )
        return LogicalSourceCounts(
            counts=counts,
            basis=basis,
            nothing_filed=frozenset(nothing),
            carried=frozenset(carried),
            read_filings=tuple(read_filings),
        )

    def read_inventory(
        self,
        *,
        entities: tuple[str, ...],
        evidence_as_of: datetime,
        request_for: Callable[[tuple[str, ...]], AlternativeEvidenceRequest],
    ) -> dict[str, str]:
        """Read each issuer's filing index at this cutoff, once, before packing.

        One registry capture and one submissions index per issuer not yet read
        at the cutoff; no filing is fetched. Each plan passes over the filings a
        current analysis at or before the cutoff read, and is sealed where the
        packing counts it and the unit's acquisition takes it. An issuer whose
        index could not be read is returned with its reason and packed on its
        reservation -- never named as filing nothing.
        """
        source = self.resources.live_source
        if source is None:
            raise ValueError("alternative_evidence.live_source_missing")
        pending: list[tuple[str, AlternativeEvidenceRequest]] = []
        for entity in entities:
            request = request_for((entity,))
            if (
                self._inventory_read(
                    entity,
                    evidence_as_of=evidence_as_of,
                    window_days=request.source_policy.sec_recent_8k_days,
                    policy_budget=request.source_policy.maximum_documents_per_issuer,
                )
                is None
            ):
                pending.append((entity, request))
        failures: dict[str, str] = {}
        if not pending:
            return failures
        reader = source.index_reader()
        try:
            registry = reader.acquire_registry(captured_at=evidence_as_of)
        except Exception as error:  # named, and the preparation packs on reservations
            reason = f"{type(error).__name__}: {str(error)[:200]}"
            return {entity: reason for entity, _request in pending}
        self.runtime.artifacts.publish("registries", registry.registry_hash, registry)
        self._inventory_registry = (evidence_as_of, registry)
        for entity, request in pending:
            try:
                plan = reader.plan_entity_filings(
                    request=request,
                    registry=registry,
                    entity_id=entity,
                    read_earlier=frozenset(
                        self.runtime.sealed_readings.read_by(
                            entity,
                            evidence_as_of=evidence_as_of,
                            window_days=request.source_policy.sec_recent_8k_days,
                        )
                    ),
                )
            except Exception as error:  # every source failure is named, never swallowed
                failures[entity] = f"{type(error).__name__}: {str(error)[:200]}"
                continue
            read = SecFilingInventoryRead(
                plan=plan,
                read_hash=inventory_read_key(
                    entity_id=plan.entity_id,
                    evidence_as_of=plan.evidence_as_of,
                    event_window_days=plan.event_window_days,
                    policy_budget=plan.policy_budget,
                    unit_capacity=plan.unit_capacity,
                ),
            )
            self.runtime.artifacts.publish("sec-inventory-reads", read.read_hash, read)
        return failures

    def _inventory_read(
        self, entity: str, *, evidence_as_of: datetime, window_days: int, policy_budget: int
    ) -> SecFilingInventoryRead | None:
        key = inventory_read_key(
            entity_id=entity,
            evidence_as_of=evidence_as_of,
            event_window_days=window_days,
            policy_budget=policy_budget,
            unit_capacity=ADMITTED_DOCUMENT_CAPACITY,
        )
        if not self.runtime.artifacts.exists("sec-inventory-reads", key):
            return None
        return self.runtime.artifacts.load("sec-inventory-reads", key, SecFilingInventoryRead)

    def _sealed_selection_plans(self) -> dict[str, list[SecFilingSelectionPlan]]:
        """Every sealed selection plan of this store, by issuer."""
        root = self.runtime.artifacts.root / "sec-selection-plans"
        plans: dict[str, list[SecFilingSelectionPlan]] = {}
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            try:
                plan = self.runtime.artifacts.load(
                    "sec-selection-plans", path.stem, SecFilingSelectionPlan
                )
            except (ValueError, FileNotFoundError):
                continue
            plans.setdefault(plan.entity_id, []).append(plan)
        return plans

    def run_binding_hash(self, *, prepare_only: bool) -> str:
        """Return the resource binding for a coverage run.

        Preparation binds only the source package; analysis also binds the
        analyst process, as a single request does.
        """
        return (
            self.resources.preparation_binding_hash if prepare_only else self.resources.binding_hash
        )

    def stale_authority(self, task: TaskRecord) -> str | None:
        """The refusal a coverage run's packets meet once their authority moved (V547).

        The Host no longer holds the authority they were prepared under -- an install replaced
        the package, or official acquisition was served or withdrawn -- so none of them can be
        read or answered.

        Args:
            task: The coverage run's Task.

        Returns:
            The refusal naming what moved; None while the run's authority holds, or for a Task
            that is not a coverage run.
        """
        if not self._is_run(task) or self._run_is_bound(task):
            return None
        return _authority_moved(self.resources, self._run(task).units[0].request.mode)

    def _run_is_bound(self, task: TaskRecord) -> bool:
        expected = self.run_binding_hash(
            prepare_only=bool(task.input.payload.get("prepare_only", True))
        )
        return bool(task.input.payload.get("resource_binding_hash") == expected)

    def admit_run(self, run: AlternativeEvidenceCoverageRun) -> None:
        """Seal the run where its Task will read it, before that Task is admitted."""
        # The preparation binding first: a run that only prepares never
        # consults the analyst process, which an un-credentialled host has
        # no business touching.
        if run.resource_binding_hash != self.resources.preparation_binding_hash and (
            run.resource_binding_hash != self.resources.binding_hash
        ):
            raise ValueError("alternative_evidence.coverage_run_binding_invalid")
        self.runtime.artifacts.publish(COVERAGE_RUN_CATEGORY, run.run_hash, run)
        self._runs[run.run_hash] = run
        if not run.units:
            held = self._carried_index().setdefault(run.scope_hash, [])
            if all(value.run_hash != run.run_hash for value in held):
                held.append(run)

    def carried_runs(self, scope_hash: str) -> tuple[AlternativeEvidenceCoverageRun, ...]:
        """Return sealed carried runs for a scope, latest cutoff first.

        These runs have nothing left to read and need no executing Task.
        """
        return tuple(
            sorted(
                self._carried_index().get(scope_hash, ()),
                key=lambda value: (value.evidence_as_of, value.run_hash),
                reverse=True,
            )
        )

    def _carried_index(self) -> dict[str, list[AlternativeEvidenceCoverageRun]]:
        if self._carried_runs is None:
            # A run a Task executes has units; only the others are read (X2).
            executed = {
                str(task.input.payload["run_hash"])
                for task in self.registry.tasks()
                if task.task_kind == self.task_kind and self._is_run(task)
            }
            index: dict[str, list[AlternativeEvidenceCoverageRun]] = {}
            root = self.runtime.artifacts.root / COVERAGE_RUN_CATEGORY
            for path in sorted(root.glob("*.json")) if root.is_dir() else ():
                if path.stem in executed:
                    continue
                try:
                    run = self.runtime.artifacts.load(
                        COVERAGE_RUN_CATEGORY, path.stem, AlternativeEvidenceCoverageRun
                    )
                except (ValueError, KnowledgeRetrievalError, FileNotFoundError):
                    continue
                if not run.units:
                    index.setdefault(run.scope_hash, []).append(run)
            self._carried_runs = index
        return self._carried_runs

    def run_of(self, task: TaskRecord) -> AlternativeEvidenceCoverageRun | None:
        """Return the coverage run named by a Task, if any.

        Read its identity regardless of the package it bound or whether it
        remains executable here.
        """
        if not self._is_run(task):
            return None
        try:
            return self._run(task)
        except (ValueError, KnowledgeRetrievalError):
            return None

    def recorded_run(self, run_hash: str) -> AlternativeEvidenceCoverageRun | None:
        """Return a run sealed by this store under ``run_hash``, if any.

        A request may name a hash for which this store has no run.
        """
        run = self._runs.get(run_hash)
        if run is not None:
            return run
        try:
            run = self.runtime.artifacts.load(
                COVERAGE_RUN_CATEGORY, run_hash, AlternativeEvidenceCoverageRun
            )
        except (ValueError, KnowledgeRetrievalError, FileNotFoundError):
            return None
        self._runs[run_hash] = run
        return run

    def completed_unit(
        self,
        intent: str,
        *,
        now: datetime | None = None,
        exclude: UUID | None = None,
        evidence_as_of: datetime | None = None,
    ) -> tuple[UUID, str | None] | None:
        """Where this preparation intent was completed, if anywhere.

        A single preparation that succeeded, or a unit of any coverage run --
        completed, cancelled or interrupted -- whose six preparation stages
        verified with their own artifacts; the intent is the identity, so the
        sealed artifacts are the same wherever they were made. With `now`,
        only a completion whose packet has not expired, read from its one
        snapshot artifact; the packet reader still verifies every byte when
        the packet is used. The intent's request names its cutoff: given it,
        a run at another cutoff is passed over by the cutoff its Task names,
        its run never read (X2).
        """
        for candidate in sorted(
            self.registry.tasks(), key=lambda value: value.updated_at, reverse=True
        ):
            if candidate.task_id == exclude or candidate.task_kind != TASK_KIND:
                continue
            purpose = candidate.input.payload.get("purpose")
            found: tuple[UUID, str | None] | None = None
            if purpose == "PREPARE_PACKET":
                if (
                    candidate.lifecycle is TaskLifecycle.SUCCEEDED
                    and preparation_intent_of(candidate) == intent
                ):
                    found = (candidate.task_id, None)
            elif purpose == COVERAGE_RUN_PURPOSE and self._run_is_bound(candidate):
                if (
                    evidence_as_of is not None
                    and datetime.fromisoformat(str(candidate.input.payload["evidence_as_of"]))
                    != evidence_as_of
                ):
                    continue
                run = self.run_of(candidate)
                if run is None:
                    continue
                matching = [unit for unit in run.units if unit.preparation_intent_hash == intent]
                if not matching:
                    continue
                states = self.unit_states(candidate)
                for unit in matching:
                    if states[unit.unit_id]["state"] == "PREPARED":
                        found = (candidate.task_id, unit.unit_id)
                        break
            if found is None or not self._prepared_under_current_bindings(candidate, found[1]):
                continue
            if now is not None:
                try:
                    _cutoff, expires_at = self.preparation_window(candidate, found[1])
                except (ValueError, KnowledgeRetrievalError):
                    continue
                if now > expires_at:
                    continue
            return found
        return None

    def _prepared_under_current_bindings(self, task: TaskRecord, unit_id: str | None) -> bool:
        """Whether a completed preparation's artifacts were sealed under the
        runtime's current acquisition, canonicalization and retrieval
        bindings. A preparation sealed under a binding this code has
        superseded is history: an analysis over it could not be published
        (`analysis_publication_binding_mismatch`), so it is neither offered
        as prepared nor copied into a new run -- the intent is the same, the
        preparation is not this code's. Seen on a QA copy across two rotations
        in one stage: the reused unit blocked every later analysis by name.
        """
        try:
            observed = (
                self._snapshot(task, unit_id).acquisition_binding_hash,
                self._document_set(task, unit_id).canonicalization_binding_hash,
                self._generation(task, unit_id).retrieval_binding_hash,
            )
        except (ValueError, KeyError, KnowledgeRetrievalError):
            return False
        installed = (
            self.runtime.acquisition.acquisition_binding_hash,
            self.runtime.document_binding_hash,
            self.runtime.retrieval_binding_hash,
        )
        return all(
            is_current(role, value, current)
            for role, value, current in zip(BINDING_ROLES[:3], observed, installed, strict=True)
        )

    def unit_states(self, task: TaskRecord) -> dict[str, dict[str, object]]:
        """Where each unit of a coverage Task stands, from its stage receipts alone.

        `prepared` when its six preparation stages verified with their own
        artifacts; `failed` with the owner's code when a stage sealed a unit
        failure; `pending` otherwise, with the stages done so far. Read by the
        progress projection and the packet readers; never inferred from the
        Task's lifecycle, which says nothing about one unit.
        """
        run = self._run(task)
        receipts = {
            value.stage_id: value.evidence[0]
            for value in self._stage_receipts(task)
            if value.evidence
        }
        prepare_only = bool(task.input.payload.get("prepare_only", True))
        expected = tuple(value[0] for value in (_STAGES[:6] if prepare_only else _STAGES))
        states: dict[str, dict[str, object]] = {}
        for unit in run.units:
            done = 0
            failure: AlternativeEvidenceUnitFailure | None = None
            for stage in expected:
                item = receipts.get(f"{unit.unit_id}_{stage}")
                if item is None:
                    break
                if item.evidence_kind == UNIT_FAILURE_KIND:
                    failure = self.runtime.artifacts.load(
                        UNIT_FAILURE_CATEGORY, item.content_hash, AlternativeEvidenceUnitFailure
                    )
                    break
                done += 1
            state: dict[str, object] = {
                "unit_id": unit.unit_id,
                "ordered_entity_ids": unit.ordered_entity_ids,
                "stages_done": done,
                "stages_expected": len(expected),
                "state": "PREPARED"
                if done >= 6
                else "FAILED"
                if failure is not None
                else "PENDING",
                "published": done >= len(expected) and not prepare_only,
                "failure_code": None if failure is None else failure.failure_code,
                "failed_stage": None if failure is None else failure.stage_id,
                "uncovered_entity_ids": () if failure is None else failure.uncovered_entity_ids,
            }
            states[unit.unit_id] = state
        return states

    def unit_authority(self, task: TaskRecord, unit_id: str | None) -> UnitAuthority:
        """Return the request, admission, and obligation for a Task stage.

        A run's unit carries the run's flags and the Task's own admission
        time, so a submission that answers the unit's packet is admitted with
        the very admission the unit was prepared under.
        """
        if self._is_run(task):
            if unit_id is None:
                raise ValueError("alternative_evidence.coverage_unit_required")
            run = self._run(task)
            if not self._run_is_bound(task):
                raise ValueError(_authority_moved(self.resources, run.units[0].request.mode))
            try:
                unit = run.unit(unit_id)
            except KeyError as error:
                raise ValueError("alternative_evidence.coverage_unit_unknown") from error
            return UnitAuthority(
                unit_id=unit_id,
                request=unit.request,
                admission=self._unit_admission(run, unit, admitted_at=task.admitted_at),
                obligation=unit.obligation,
            )
        if unit_id is not None:
            raise ValueError("alternative_evidence.coverage_unit_unknown")
        request, admission = self._authority(task)
        return UnitAuthority(
            unit_id=None, request=request, admission=admission, obligation=self._obligation(task)
        )

    @staticmethod
    def _unit_admission(
        run: AlternativeEvidenceCoverageRun,
        unit: AlternativeEvidenceCoverageUnit,
        *,
        admitted_at: datetime,
    ) -> AlternativeEvidenceAdmission:
        """The Host permission one unit's request carries: the run's flags, the
        unit's request, the Task's own admission time.
        """
        return seal_contract(
            AlternativeEvidenceAdmission,
            "admission_hash",
            request_hash=unit.request.request_hash,
            network_consent=run.network_consent,
            admit_live_official=run.admit_live_official,
            admit_model_review=run.admit_model_review,
            admitted_at=admitted_at,
        )

    def _unit_failure_evidence(self, task: TaskRecord, unit_id: str) -> TaskEvidence | None:
        for value in self._stage_receipts(task):
            head, _stage = split_unit_stage(value.stage_id)
            if (
                head == unit_id
                and value.evidence
                and (value.evidence[0].evidence_kind == UNIT_FAILURE_KIND)
            ):
                return value.evidence[0]
        return None

    def _completed_unit_source(
        self, task: TaskRecord, authority: UnitAuthority
    ) -> tuple[UUID, str | None] | None:
        """Where this unit's intent was already prepared, if anywhere.

        A single preparation that succeeded, or a unit of any coverage run --
        completed, cancelled or interrupted -- whose six preparation stages
        verified with their own artifacts. The sealed artifacts are the same
        either way; the intent is the identity, and a completion never
        changes, so it is remembered for the life of this adapter.
        """
        intent = preparation_intent_hash(
            request=authority.request,
            obligation=authority.obligation,
            admission=authority.admission,
            resource_binding_hash=str(task.input.payload["resource_binding_hash"]),
        )
        found = self._completed_units.get(intent)
        if found is None and (intent, task.task_id) not in self._absent_units:
            found = self.completed_unit(
                intent, exclude=task.task_id, evidence_as_of=authority.request.evidence_as_of
            )
            if found is not None:
                self._completed_units[intent] = found
            else:
                self._absent_units.add((intent, task.task_id))
        return found

    def _copied_evidence(self, source: tuple[UUID, str | None], stage: str) -> TaskEvidence | None:
        """A completed unit's own stage evidence, carried into this run as is."""
        if stage not in _PREPARATION_STAGES:
            return None
        task_id, unit_id = source
        prefix = "" if unit_id is None else f"{unit_id}_"
        for value in self._stage_receipts(self.registry.task(task_id)):
            if value.stage_id == f"{prefix}{stage}" and len(value.evidence) == 1:
                item = value.evidence[0]
                if item.evidence_kind == UNIT_FAILURE_KIND:
                    return None
                return item
        return None

    ANALYSIS_PROCEDURE = (
        "Extract cited facts and counterevidence; "
        "do not assign portfolio severity or change positions."
    )

    def analysis_context(
        self,
        task_id: UUID,
        *,
        now: datetime,
        unit_id: str | None = None,
        span_handles: tuple[str, ...] | None = None,
        delivery_part: tuple[int, int] | None = None,
    ) -> tuple[AlternativeEvidencePacket, dict[str, object]]:
        """Project the existing Analyst contract over an exact prepared packet.

        `span_handles` renders one delivery part of the packet; the context
        hash is the whole packet's -- its identity, the schema, the policies
        and the expiry -- so every part of one packet binds the same answer,
        whatever budget it was delivered under.
        """
        packet = self.prepared_packet(task_id, now=now, unit_id=unit_id)
        schema = AlternativeEvidenceAnalystAnswer.model_json_schema()
        fields = schema["$defs"]["AlternativeEvidenceAnswerFinding"]["properties"]
        fields["issuer"]["enum"] = list(packet.request.ordered_entity_ids)
        aliases = list(span_aliases(packet.spans, packet.request.ordered_entity_ids))
        for key in ("cite", "contrary"):
            fields[key]["items"]["enum"] = aliases
        if not packet.spans:
            schema["properties"]["findings"]["maxItems"] = 0
        basis: dict[str, object] = {
            "prepared_task_id": str(task_id),
            **({} if unit_id is None else {"prepared_unit_id": unit_id}),
            "packet_hash": _hash(packet),
            "analysis_policy_hash": self.runtime.analysis_policy_hash,
            "decision_policy_hash": self.runtime.decision_policy.binding_hash,
            "response_schema": schema,
            "source_expires_at": packet.snapshot.expires_at.isoformat(),
            "task_procedure": self.ANALYSIS_PROCEDURE,
        }
        # The Analyst reads the schema's words; the context binds its structure (SC3).
        context_hash = _hash({**basis, "response_schema": schema_structure(schema)})
        body: dict[str, object] = {
            **{key: value for key, value in basis.items() if key != "task_procedure"},
            "packet": render_evidence_packet(
                packet,
                span_handles=span_handles,
                delivery_part=delivery_part,
            ),
        }
        return packet, {**body, "analysis_context_hash": context_hash}

    def preparation_window(
        self, task: TaskRecord, unit_id: str | None = None
    ) -> tuple[datetime, datetime]:
        """Return a completed preparation's cutoff and packet expiration.

        One artifact read, so a state projection can say which packet is
        prepared and until when without re-verifying the packet; the packet
        reader still verifies every byte and binding when the packet is used.
        """
        authority = self.unit_authority(task, unit_id)
        self._require_prepared(task, unit_id)
        if not self._prepared_under_current_bindings(task, unit_id):
            # Built under an acquisition, canonicalization or retrieval
            # contract this workspace has superseded: an analysis against it
            # could not be sealed, so it is not a prepared packet to offer,
            # only history.
            raise ValueError("alternative_evidence.preparation_superseded")
        return authority.request.evidence_as_of, self._snapshot(task, unit_id).expires_at

    def source_check(
        self, task_id: UUID, *, unit_id: str | None = None
    ) -> dict[str, object] | None:
        """Return source check facts from a Task's acquisition snapshot.

        Include the check time, metadata and body traffic, and each resource's
        outcome and byte count. Return ``None`` if no snapshot was sealed, if
        it is a recorded snapshot, or if it predates this accounting.
        """
        task = self.registry.task(task_id)
        snapshot = self._snapshot(task, unit_id)
        accounting = snapshot.acquisition
        if accounting is None:
            return None
        return {
            "checked_at": snapshot.published_at.isoformat(),
            "evidence_as_of": self.unit_authority(task, unit_id).request.evidence_as_of.isoformat(),
            "snapshot_status": snapshot.status.value,
            "inventory_request_count": accounting.inventory_request_count,
            "history_shard_request_count": accounting.history_shard_request_count,
            "body_request_count": accounting.body_request_count,
            "reused_local_count": accounting.reused_local_count,
            "fetched_count": accounting.fetched_count,
            "deferred_count": accounting.deferred_count,
            "failed_count": accounting.failed_count,
            "fetched_bytes": accounting.fetched_bytes,
            "reused_bytes": accounting.reused_bytes,
            "documents": [value.model_dump(mode="json") for value in accounting.documents],
        }

    def _require_prepared(self, task: TaskRecord, unit_id: str | None) -> None:
        """A single preparation must have succeeded; a run's unit must have its
        six stages verified, whatever became of the run around it.
        """
        if unit_id is None:
            if (
                task.input.payload.get("purpose") not in _PREPARED_PURPOSES
                or task.lifecycle is not TaskLifecycle.SUCCEEDED
            ):
                raise ValueError("alternative_evidence.preparation_not_complete")
            return
        state = self.unit_states(task).get(unit_id)
        if state is None:
            raise ValueError("alternative_evidence.coverage_unit_unknown")
        if state["state"] == "FAILED":
            raise ValueError(f"alternative_evidence.unit_not_prepared:{state['failure_code']}")
        if state["state"] != "PREPARED":
            raise ValueError("alternative_evidence.preparation_not_complete")

    def run_tasks(self, run: AlternativeEvidenceCoverageRun) -> tuple[TaskRecord, ...]:
        """Return the SUCCEEDED Tasks that executed exactly this run, the newest first.

        Args:
            run: The coverage run.

        Returns:
            Its succeeded Tasks, newest first.
        """
        return tuple(
            sorted(
                (
                    task
                    for task in self.registry.tasks()
                    if task.task_kind == self.task_kind
                    and task.lifecycle is TaskLifecycle.SUCCEEDED
                    and task.input.payload.get("purpose") == COVERAGE_RUN_PURPOSE
                    and task.input.payload.get("run_hash") == run.run_hash
                ),
                key=lambda task: (task.updated_at, str(task.task_id)),
                reverse=True,
            )
        )

    def completed_run(
        self, run: AlternativeEvidenceCoverageRun, *, now: datetime
    ) -> TaskRecord | None:
        """Return the newest SUCCEEDED Task of this run whose every unit's packet reads now.

        Args:
            run: The coverage run.
            now: The moment the packets are read at.

        Returns:
            That Task, or None.
        """
        for task in self.run_tasks(run):
            try:
                for unit in run.units:
                    self.prepared_packet(task.task_id, now=now, unit_id=unit.unit_id)
            except (ValueError, KnowledgeRetrievalError):
                continue
            return task
        return None

    def run_expired(self, run: AlternativeEvidenceCoverageRun, *, now: datetime) -> bool:
        """Say whether a Task of this run prepared every unit and only its packets expired.

        Args:
            run: The coverage run.
            now: The moment the packets are read at.

        Returns:
            True when a Task prepared every unit and each packet refuses only as stale.
        """
        for task in self.run_tasks(run):
            states = self.unit_states(task)
            if any(value["state"] != "PREPARED" for value in states.values()):
                continue
            stale = False
            for unit in run.units:
                try:
                    self.prepared_packet(task.task_id, now=now, unit_id=unit.unit_id)
                except ValueError as error:
                    if str(error) == "alternative_evidence.brief_source_stale":
                        stale = True
                        continue
                    stale = False
                    break
                except KnowledgeRetrievalError:
                    stale = False
                    break
            if stale:
                return True
        return False

    def binding_changes(
        self, claimed: str, run: AlternativeEvidenceCoverageRun, *, policy: AdmittedEvidencePolicy
    ) -> tuple[str, ...]:
        """Name what moved between a captured run and the run this workspace forms now.

        Args:
            claimed: The captured run's hash.
            run: The run formed now.
            policy: The admitted evidence policy both runs are read under.

        Returns:
            Each part that moved (cutoff, issuer scope, source package, evidence policy,
            matter selection, permissions), once, in that order; none when the captured run
            is not recorded.
        """
        recorded = self.recorded_run(claimed)
        if recorded is None:
            return ()
        moved: list[str] = []
        if recorded.evidence_as_of != run.evidence_as_of:
            moved.append("cutoff")
        if recorded.covered_entity_ids != run.covered_entity_ids:
            moved.append("issuer_scope")
        if recorded.resource_binding_hash != run.resource_binding_hash:
            moved.append("source_package")
        theirs, ours = policy.run_request(recorded), policy.run_request(run)
        if (
            theirs.source_policy != ours.source_policy
            or theirs.evidence_classes != ours.evidence_classes
            or theirs.mode is not ours.mode
            or theirs.ttl_seconds != ours.ttl_seconds
        ):
            moved.append("evidence_policy")
        if theirs.matter_selection != ours.matter_selection:
            moved.append("matter_selection")
        if (
            recorded.network_consent,
            recorded.admit_live_official,
            recorded.admit_model_review,
            recorded.unit_limit,
        ) != (
            run.network_consent,
            run.admit_live_official,
            run.admit_model_review,
            run.unit_limit,
        ):
            moved.append("permissions")
        return tuple(dict.fromkeys(moved))

    def source_inventory(
        self,
        run: AlternativeEvidenceCoverageRun,
        recorded: tuple[RecordedEvidenceDocument, ...],
        *,
        mode: AlternativeEvidenceMode,
        ciks: Mapping[str, str],
    ) -> dict[str, object]:
        """Say what the admitted source holds for every issuer of the run at its cutoff.

        Unit by unit: the documents before the cutoff, the newest one, what each unit's
        capacity admits and defers, and the issuers with no source at all -- the
        source-coverage denominator the preparation answers against, read from the recorded
        source owner. A live source reports this workspace's holdings instead.

        Args:
            run: The coverage run.
            recorded: The recorded documents of the run's issuers.
            mode: The admitted source mode.
            ciks: Each registered issuer's CIK by entity id.

        Returns:
            The inventory, by issuer and by unit.
        """
        if mode is not AlternativeEvidenceMode.RECORDED:
            return self.local_holdings(run, mode=mode, ciks=ciks)
        source = self.runtime.acquisition.recorded
        issuers: list[dict[str, object]] = []
        units: list[dict[str, object]] = []
        without: list[str] = []
        for unit in run.units:
            selected, deferrals = source.select_documents(request=unit.request, documents=recorded)
            by_entity: dict[str, list[RecordedEvidenceDocument]] = {
                entity: [] for entity in unit.ordered_entity_ids
            }
            for _sequence, document in selected:
                by_entity[document.entity_id].append(document)
            capacity = source.capacity(unit.request)
            units.append(
                {
                    "unit_id": unit.unit_id,
                    "capacity_per_issuer": capacity,
                    "capacity_basis": "the policy budget per issuer, the admitted document set "
                    "per unit; never the set divided by the unit's issuers",
                    "documents_selected": len(selected),
                    "deferrals": list(deferrals),
                }
            )
            for entity in unit.ordered_entity_ids:
                documents = by_entity[entity]
                if not documents:
                    without.append(entity)
                issuers.append(
                    {
                        "entity_id": entity,
                        "unit_id": unit.unit_id,
                        "documents": len(documents),
                        "document_types": sorted({value.document_type for value in documents}),
                        "latest_available_at": (
                            max(value.available_at for value in documents).isoformat()
                            if documents
                            else None
                        ),
                    }
                )
        return {
            "mode": mode.value,
            "issuers": issuers,
            "issuers_with_source": len(issuers) - len(without),
            "issuers_without_source": without,
            "units": units,
        }

    def local_holdings(
        self,
        run: AlternativeEvidenceCoverageRun,
        *,
        mode: AlternativeEvidenceMode,
        ciks: Mapping[str, str],
    ) -> dict[str, object]:
        """Say what this workspace already holds for the run's issuers.

        Retained SEC bodies by issuer, from the sealed commitments alone: counts and bytes as
        their references state them, verified against the bytes only at use. Nothing here
        reads the source or grants freshness: the next preparation checks the official
        inventory and reuses every held body it still selects.

        Args:
            run: The coverage run.
            mode: The admitted source mode.
            ciks: Each registered issuer's CIK by entity id.

        Returns:
            The holdings by issuer, with the bytes held and the documents deferred.
        """
        issuers: list[dict[str, object]] = []
        held_bytes = 0
        deferred = self.runtime.local_sources.deferrals()
        for unit in run.units:
            for entity in unit.ordered_entity_ids:
                cik = ciks.get(entity)
                held = self.runtime.local_sources.holdings(entity_id=entity, cik=cik)
                held_bytes += sum(value.content_bytes for value in held)
                issuers.append(
                    {
                        "entity_id": entity,
                        "unit_id": unit.unit_id,
                        "cik": cik,
                        "retained_documents": len(held),
                        "retained_bytes": sum(value.content_bytes for value in held),
                        # Resources observed oversize and sealed as deferred:
                        # visibly unavailable, never an empty success.
                        "deferred_documents": [
                            {
                                "form": value.form,
                                "accession": value.accession,
                                "observed_bytes": value.observed_bytes,
                                "admitted_cap_bytes": value.admitted_cap_bytes,
                                # No time-based recheck since W6: a larger
                                # cap or an explicit scope retries it.
                                "recheck_after": None,
                            }
                            for value in deferred
                            if value.source_cik == cik
                        ],
                        "document_types": sorted({value.document_type for value in held}),
                        "latest_accepted_at": (
                            max(
                                value.accepted_at for value in held if value.accepted_at is not None
                            ).isoformat()
                            if any(value.accepted_at is not None for value in held)
                            else None
                        ),
                    }
                )
        return {
            "mode": mode.value,
            "issuers": issuers,
            "issuers_with_retained_documents": sum(
                1 for value in issuers if cast(int, value["retained_documents"]) > 0
            ),
            "retained_bytes": held_bytes,
            "units": [],
            "claim": (
                "Retained bodies are reused without a download when the official "
                "inventory still selects them; they are verified against their "
                "references at use, never assumed current."
            ),
        }

    def reuse_view(self, prepared: list[tuple[str, str | None]]) -> dict[str, object] | None:
        """Say how the named preparations reused sealed work.

        Read from their own sealed receipts and source checks: which selections were reused
        whole from an earlier receipt (and which), and how many retained documents were
        reused or fetched. The service's own counters ride beside them, labelled as what they
        are: totals since this service started, across every preparation.

        Args:
            prepared: Each prepared Task id with its unit id.

        Returns:
            The reuse by preparation, or None when nothing is prepared.
        """
        if not prepared:
            return None
        rows: list[dict[str, object]] = []
        reused_documents = fetched_documents = 0
        checked = False
        for task_id, unit_id in prepared:
            try:
                receipt_hash, _span_set = self.prepared_receipt_identity(
                    UUID(task_id), unit_id=unit_id
                )
                receipt = self.runtime.artifacts.load(
                    "retrieval-access-receipts",
                    receipt_hash,
                    AlternativeEvidenceRetrievalAccessReceipt,
                )
            except (ValueError, KeyError, KnowledgeRetrievalError):
                rows.append({"task_id": task_id, "unit_id": unit_id, "selection": "UNREADABLE"})
                continue
            rows.append(
                {
                    "task_id": task_id,
                    "unit_id": unit_id,
                    "selection": "REUSED" if receipt.reused_from_receipt_hash else "SELECTED",
                    "reused_from_receipt_hash": receipt.reused_from_receipt_hash,
                }
            )
            check = self.prepared_source_check(task_id, unit_id)
            if check is not None:
                checked = True
                reused_documents += int(cast("int | None", check.get("reused_local_count")) or 0)
                fetched_documents += int(cast("int | None", check.get("fetched_count")) or 0)
        return {
            "preparations": rows,
            "selections_reused": sum(1 for row in rows if row["selection"] == "REUSED"),
            "selections_made": sum(1 for row in rows if row["selection"] == "SELECTED"),
            "documents_reused": reused_documents if checked else None,
            "documents_fetched": fetched_documents if checked else None,
            "service_counters": self.runtime.reuse_accounting(),
            "basis": (
                "Selections and documents are this preparation's own sealed receipts and "
                "source checks; service_counters are totals since this service started, "
                "across every preparation, not this one's."
            ),
        }

    def prepared_source_check(
        self, task_id: str | None, unit_id: str | None
    ) -> dict[str, object] | None:
        """Return the source check a prepared Task performed, from its own snapshot.

        The traffic (inventory, history and body requests) and the outcome of every resource
        -- reused, fetched, deferred, failed -- with the bytes each way.

        Args:
            task_id: The prepared Task's id, or None.
            unit_id: The unit, or None for a single preparation.

        Returns:
            The check, or None for a recorded preparation, one sealed before the accounting
            existed, or an id that names no prepared Task.
        """
        if task_id is None:
            return None
        try:
            return self.source_check(UUID(task_id), unit_id=unit_id)
        except (ValueError, KeyError, KnowledgeRetrievalError):
            return None

    def prepared_receipts(self) -> dict[str, tuple[str, str | None]]:
        """Return every sealed preparation receipt this workspace's succeeded Tasks hold.

        Returns:
            By receipt hash, the Task and unit that sealed it (the first found).
        """
        found: dict[str, tuple[str, str | None]] = {}
        for task in self.registry.tasks():
            if task.task_kind != self.task_kind or task.lifecycle is not TaskLifecycle.SUCCEEDED:
                continue
            units: tuple[str | None, ...] = (None,)
            if task.input.payload.get("purpose") == COVERAGE_RUN_PURPOSE:
                try:
                    states = self.unit_states(task)
                except (ValueError, KeyError):
                    continue
                units = tuple(u for u, state in states.items() if state["state"] == "PREPARED")
            for unit_id in units:
                try:
                    receipt_hash, _span_set = self.prepared_receipt_identity(
                        task.task_id, unit_id=unit_id
                    )
                except (ValueError, KeyError, KnowledgeRetrievalError):
                    continue
                found.setdefault(receipt_hash, (str(task.task_id), unit_id))
        return found

    def prepared_packet(
        self, task_id: UUID, *, now: datetime, unit_id: str | None = None
    ) -> AlternativeEvidencePacket:
        """Reopen completed preparation, verifying source bytes without re-embedding.

        A preparation sealed under a binding this code has superseded is
        refused by that name (`preparation_superseded`, as the continuation
        refuses it), never as a lineage mismatch: its handles stay readable
        through the publications sealed over it, and the next step is to
        prepare again under the current contract.
        """
        task = self.registry.task(task_id)
        authority = self.unit_authority(task, unit_id)
        request = authority.request
        self._require_prepared(task, unit_id)
        if not self._prepared_under_current_bindings(task, unit_id):
            raise ValueError("alternative_evidence.preparation_superseded")
        snapshot = self._snapshot(task, unit_id)
        if now > snapshot.expires_at:
            raise ValueError("alternative_evidence.brief_source_stale")
        document_set = self._document_set(task, unit_id)
        generation = self._generation(task, unit_id)
        receipt, spans = self._packet_inputs(task, unit_id)
        if (
            snapshot.request_hash != request.request_hash
            or document_set.request_hash != request.request_hash
            or document_set.source_snapshot_hash != snapshot.snapshot_hash
            or generation.document_set_hash != document_set.document_set_hash
            or receipt.request_hash != request.request_hash
            or receipt.document_set_hash != document_set.document_set_hash
            or receipt.retrieval_generation_hash != generation.generation_hash
            or set(receipt.delivered_span_handles) != {span.span_handle for span in spans}
        ):
            raise ValueError("alternative_evidence.brief_lineage_mismatch")
        # A document whose Workspace blob no longer matches its revision is a
        # refusal of this packet, named as such: the kernel's own failure code
        # travels in the refusal instead of escaping as an untyped error.
        try:
            self.runtime.documents.verify(document_set)
        except KnowledgeRetrievalError as error:
            raise ValueError(
                f"alternative_evidence.brief_document_unverifiable:{error.failure.code}"
            ) from error
        references = {value.semantic_handle: value for value in document_set.documents}
        contents: dict[str, bytes] = {}
        original_for: Callable[[str], tuple[bytes, str, str] | None] | None = None
        for span in spans:
            reference = references.get(span.document_handle)
            if reference is None or (
                span.entity_id,
                span.revision_label,
                span.document_type,
                span.available_at,
            ) != (
                reference.entity_id,
                reference.revision_label,
                reference.document_type,
                reference.available_at,
            ):
                raise ValueError("alternative_evidence.brief_document_lineage_invalid")
            if span.document_handle not in contents:
                try:
                    _, contents[span.document_handle] = (
                        self.runtime.documents.library.read_revision(
                            reference.workspace_document_id, reference.workspace_revision
                        )
                    )
                except KnowledgeRetrievalError as error:
                    raise ValueError(
                        f"alternative_evidence.brief_document_unverifiable:{error.failure.code}"
                    ) from error
            content = contents[span.document_handle]
            text = content.decode("utf-8")
            anchored = text[span.character_start : span.character_end]
            # A table view's range is the placeholder line, verified as any
            # span's; its excerpt is the page rendered from the retained
            # original, verified by rendering it again.
            excerpt = span.excerpt if span.table_view is None else anchored
            if (
                anchored != excerpt
                or content[span.utf8_byte_start : span.utf8_byte_end] != excerpt.encode("utf-8")
                or len(text[: span.character_start].encode("utf-8")) != span.utf8_byte_start
                or text.count("\n", 0, span.character_start) + 1 != span.start_line
                or text.count("\n", 0, span.character_end) + 1 != span.end_line
            ):
                raise ValueError("alternative_evidence.brief_span_source_mismatch")
            if span.table_view is not None:
                if original_for is None:
                    original_for = self.runtime.original_reader(document_set)
                _verify_table_view(
                    span,
                    text=text,
                    reference=reference,
                    original=original_for(reference.workspace_document_id),
                )
        return AlternativeEvidencePacket(
            request=request,
            obligation=authority.obligation,
            snapshot=snapshot,
            document_set=document_set,
            receipt=receipt,
            spans=spans,
        )

    def validate_submission(
        self,
        submitted: SubmittedEvidenceAnalysis,
        *,
        now: datetime,
        packet: AlternativeEvidencePacket | None = None,
    ) -> AlternativeEvidenceAnalystAnswer:
        """Validate a submitted answer through the neutral sealer.

        Use this before admission and again at execution. ``packet`` is the one the
        admitting request already verified (V117); at execution it is opened
        and verified again.
        """
        answer = require_answer_format(submitted.answer)
        if packet is None:
            packet = self.prepared_packet(
                submitted.prepared_task_id, now=now, unit_id=submitted.prepared_unit_id
            )
        if (
            submitted.packet_hash != _hash(packet)
            or not is_current(
                ANALYSIS_POLICY_ROLE,
                submitted.analysis_policy_hash,
                self.runtime.analysis_policy_hash,
            )
            or not is_current(
                DECISION_POLICY_ROLE,
                submitted.decision_policy_hash,
                self.runtime.decision_policy.binding_hash,
            )
        ):
            raise ValueError("alternative_evidence.external_analysis_binding_changed")
        seal_alternative_evidence_analyst_brief(
            request=packet.request,
            obligation=packet.obligation,
            snapshot=packet.snapshot,
            document_set=packet.document_set,
            generation=self._generation(
                self.registry.task(submitted.prepared_task_id), submitted.prepared_unit_id
            ),
            access_receipt=packet.receipt,
            resolved_spans=packet.spans,
            analysis_policy=self.runtime.analysis_policy,
            decision_policy=self.runtime.decision_policy,
            playpen_root=self.runtime.playpen_root,
            answer=answer,
            dropped=submitted.dropped,
            completed_at=now,
            actor_kind=submitted.actor_submission.actor_kind,
            actor_id=submitted.actor_submission.actor_id,
        )
        return answer

    @staticmethod
    def _continuation(task: TaskRecord) -> EvidenceContinuation:
        value: EvidenceContinuation = EvidenceContinuation.model_validate(
            task.input.payload["continuation"]
        )
        return value

    def prepared_receipt_identity(
        self, task_id: UUID, *, unit_id: str | None = None
    ) -> tuple[str, str]:
        """Return the identities a continuation must name.

        They identify the access receipt and resolved span set sealed by a
        prepared packet.
        """
        task = self.registry.task(task_id)
        self.unit_authority(task, unit_id)
        self._require_prepared(task, unit_id)
        evidence = self._stage_evidence(task, "select_evidence_spans", unit_id)
        return str(evidence.content_hash), _parameter(evidence.reference, "spans")

    def continuation_prior(
        self, continuation: EvidenceContinuation
    ) -> tuple[
        AlternativeEvidenceRetrievalAccessReceipt, tuple[AlternativeEvidenceResolvedSpan, ...]
    ]:
        """Verify and return the receipt and spans a continuation names.

        They must be those sealed by the prepared packet. Refuse a stale or
        unrelated receipt or a span set outside its commitment.
        """
        receipt_hash, span_set_hash = self.prepared_receipt_identity(
            continuation.prepared_task_id, unit_id=continuation.prepared_unit_id
        )
        if (
            receipt_hash != continuation.continuation_of
            or span_set_hash != continuation.continuation_spans
        ):
            raise ValueError("alternative_evidence.litigation_continuation_prior_mismatch")
        return self._packet_inputs(
            self.registry.task(continuation.prepared_task_id), continuation.prepared_unit_id
        )

    @staticmethod
    def _submission(task: TaskRecord) -> SubmittedEvidenceAnalysis:
        value: SubmittedEvidenceAnalysis = SubmittedEvidenceAnalysis.model_validate(
            task.input.payload["submitted_analysis"]
        )
        return value

    def _resolve_registry(
        self,
        *,
        request: AlternativeEvidenceRequest,
        observed_at: datetime,
        task_id: UUID | None = None,
    ) -> SecIssuerRegistrySnapshot:
        """The issuer registry this stage resolves: the recorded one, or the
        official one captured once per Task -- a run's later units reuse the
        capture its first unit made rather than fetching the same registry
        again (the official file is close to a megabyte, and a run of eight
        units fetched it eight times).
        """
        if request.mode is AlternativeEvidenceMode.RECORDED:
            value = self.resources.recorded_registry
            if value is None:
                raise ValueError("alternative_evidence.recorded_registry_missing")
        else:
            source = self.resources.live_source
            if source is None:
                raise ValueError("alternative_evidence.live_source_missing")
            held = self._task_registry
            read = self._inventory_registry
            if task_id is not None and held is not None and held[0] == task_id:
                value = held[1]
            elif read is not None and read[0] == request.evidence_as_of:
                value = read[1]
                if task_id is not None:
                    self._task_registry = (task_id, value)
            else:
                value = source.acquire_registry(captured_at=observed_at)
                if task_id is not None:
                    self._task_registry = (task_id, value)
        self.runtime.artifacts.publish("registries", value.registry_hash, value)
        return value

    def _planned(self, task: TaskRecord, unit_id: str | None, stage: str) -> int:
        """A counted stage's denominator known before it starts: the filings a
        unit's plan counts (the run's source counts) for its acquisition, one
        unit for its selection; the others learn theirs as they run.
        """
        if stage == SELECT:
            return 1
        if stage != ACQUIRE or unit_id is None or not self._is_run(task):
            return 0
        run = self._run(task)
        counts = dict(run.source_counts)
        return sum(counts.get(entity, 0) for entity in run.unit(unit_id).ordered_entity_ids)

    def _acquire(
        self,
        task: TaskRecord,
        *,
        request: AlternativeEvidenceRequest,
        admission: AlternativeEvidenceAdmission,
        unit_id: str | None = None,
        fetched: Callable[[int], None] | None = None,
    ) -> tuple[AlternativeEvidenceSnapshot, AcquiredEvidenceSourceReferenceSet]:
        registry = self._registry(task, unit_id)
        now = task.updated_at
        if request.mode is AlternativeEvidenceMode.RECORDED:
            return self.runtime.acquire_recorded(
                request=request,
                registry=registry,
                documents=self.resources.recorded_documents,
                published_at=now,
            )
        source = self.resources.live_source
        if source is None:
            raise ValueError("alternative_evidence.live_source_missing")
        _registry, snapshot, source_set = self.runtime.acquire_live(
            request=request,
            admission=admission,
            source=source,
            registry=registry,
            published_at=now,
            should_cancel=lambda: (
                self.registry.task(task.task_id).lifecycle is TaskLifecycle.CANCEL_REQUESTED
            ),
            clock=self.clock,
            fetched=fetched,
        )
        return snapshot, source_set

    def _packet_inputs(
        self, task: TaskRecord, unit_id: str | None = None
    ) -> tuple[
        AlternativeEvidenceRetrievalAccessReceipt,
        tuple[AlternativeEvidenceResolvedSpan, ...],
    ]:
        evidence = self._stage_evidence(task, "select_evidence_spans", unit_id)
        receipt = self.runtime.artifacts.load(
            "retrieval-access-receipts",
            evidence.content_hash,
            AlternativeEvidenceRetrievalAccessReceipt,
        )
        span_set = self.runtime.artifacts.load(
            "resolved-span-sets",
            _parameter(evidence.reference, "spans"),
            AlternativeEvidenceResolvedSpanSet,
        )
        return receipt, span_set.spans

    def _registry(self, task: TaskRecord, unit_id: str | None = None) -> SecIssuerRegistrySnapshot:
        return self.runtime.artifacts.load(
            "registries",
            self._stage_identity(task, "resolve_official_sources", unit_id),
            SecIssuerRegistrySnapshot,
        )

    def _source_set(
        self, task: TaskRecord, unit_id: str | None = None
    ) -> AcquiredEvidenceSourceSet:
        return self.runtime.artifacts.load_source_set(
            self._stage_identity(task, "acquire_source_evidence", unit_id)
        )

    def _snapshot(
        self, task: TaskRecord, unit_id: str | None = None
    ) -> AlternativeEvidenceSnapshot:
        source = self._stage_evidence(task, "acquire_source_evidence", unit_id)
        return self.runtime.artifacts.load(
            "snapshots", _parameter(source.reference, "snapshot"), AlternativeEvidenceSnapshot
        )

    def _document_set(
        self, task: TaskRecord, unit_id: str | None = None
    ) -> AlternativeEvidenceDocumentSet:
        return self.runtime.artifacts.load(
            "document-sets",
            self._stage_identity(task, "canonicalize_documents", unit_id),
            AlternativeEvidenceDocumentSet,
        )

    def _generation(
        self, task: TaskRecord, unit_id: str | None = None
    ) -> RetrievalGenerationRecord:
        return self.runtime.artifacts.load_retrieval_generation(
            self._stage_identity(task, "build_retrieval_generation", unit_id)
        )

    def _stage_identity(self, task: TaskRecord, stage: str, unit_id: str | None = None) -> str:
        return cast(str, self._stage_evidence(task, stage, unit_id).content_hash)

    def _stage_evidence(
        self, task: TaskRecord, stage: str, unit_id: str | None = None
    ) -> TaskEvidence:
        purpose = task.input.payload.get("purpose")
        delegated = (
            _STAGES[:6]
            if purpose == "SUBMITTED_ANALYSIS"
            else _STAGES[:5]
            if purpose == "CONTINUE_READING"
            else ()
        )
        if stage in {value[0] for value in delegated}:
            if purpose == "SUBMITTED_ANALYSIS":
                submitted = self._submission(task)
                prepared_task_id, unit_id = submitted.prepared_task_id, submitted.prepared_unit_id
            else:
                continuation = self._continuation(task)
                prepared_task_id = continuation.prepared_task_id
                unit_id = continuation.prepared_unit_id
            source = self.registry.task(prepared_task_id)
            if unit_id is None:
                if (
                    source.input.payload.get("purpose") not in _PREPARED_PURPOSES
                    or source.lifecycle is not TaskLifecycle.SUCCEEDED
                    or any(
                        source.input.payload[key] != task.input.payload[key]
                        for key in ("request", "admission", "obligation", "resource_binding_hash")
                    )
                ):
                    raise ValueError("alternative_evidence.preparation_task_mismatch")
                self._authority(source)
            else:
                # The answered packet is one unit of a coverage run: the run's
                # unit must be the request, question and package this answer
                # was admitted for, and it must have been prepared.
                if not self._is_run(source) or not source.input.payload.get("prepare_only"):
                    raise ValueError("alternative_evidence.preparation_task_mismatch")
                authority = self.unit_authority(source, unit_id)
                if (
                    authority.request.model_dump(mode="json") != task.input.payload["request"]
                    or authority.obligation.model_dump(mode="json")
                    != task.input.payload["obligation"]
                    or source.input.payload["resource_binding_hash"]
                    != task.input.payload["resource_binding_hash"]
                ):
                    raise ValueError("alternative_evidence.preparation_task_mismatch")
                self._require_prepared(source, unit_id)
            # The source may itself be a continuation whose own lineage lives
            # in the packet it continued: the chain resolves link by link.
            return self._stage_evidence(source, stage, unit_id)
        stage_id = stage if unit_id is None else f"{unit_id}_{stage}"
        matches = [value for value in self._stage_receipts(task) if value.stage_id == stage_id]
        if len(matches) != 1 or len(matches[0].evidence) != 1:
            raise ValueError("alternative_evidence.stage_lineage_missing")
        item = matches[0].evidence[0]
        if item.evidence_kind == UNIT_FAILURE_KIND:
            failure = self.runtime.artifacts.load(
                UNIT_FAILURE_CATEGORY, item.content_hash, AlternativeEvidenceUnitFailure
            )
            raise ValueError(f"alternative_evidence.unit_not_prepared:{failure.failure_code}")
        return item

    def _authority(
        self,
        task: TaskRecord,
    ) -> tuple[AlternativeEvidenceRequest, AlternativeEvidenceAdmission]:
        if task.input.input_schema_id != INPUT_SCHEMA_ID:
            raise ValueError("alternative_evidence.current_authority_mismatch")
        purpose = task.input.payload.get("purpose")
        if purpose == COVERAGE_RUN_PURPOSE:
            raise ValueError("alternative_evidence.coverage_unit_required")
        if purpose not in {None, "PREPARE_PACKET", "SUBMITTED_ANALYSIS", "CONTINUE_READING"}:
            raise ValueError("alternative_evidence.task_purpose_invalid")
        binding = (
            self.resources.binding_hash
            if purpose is None
            else self.resources.preparation_binding_hash
        )
        request = AlternativeEvidenceRequest.model_validate(task.input.payload["request"])
        if task.input.payload.get("resource_binding_hash") != binding:
            raise ValueError(_authority_moved(self.resources, request.mode))
        return (
            request,
            AlternativeEvidenceAdmission.model_validate(task.input.payload["admission"]),
        )

    @staticmethod
    def _obligation(task: TaskRecord) -> AlternativeEvidenceResearchObligation:
        """The obligation this Task was admitted with, re-validated on every read.

        The Task record is durable and a stage may run after a restart, so the
        check travels with the read rather than happening once at admission.
        """
        raw = task.input.payload.get("obligation")
        if raw is None:
            raise ValueError("alternative_evidence.task_obligation_missing")
        obligation: AlternativeEvidenceResearchObligation = (
            AlternativeEvidenceResearchObligation.model_validate(raw)
        )
        request = AlternativeEvidenceRequest.model_validate(task.input.payload["request"])
        validate_scoped_obligation(request=request, obligation=obligation)
        return obligation


def _evidence(kind: str, identity: str, **parameters: str) -> TaskEvidence:
    reference = f"semantic://alternative-evidence/{kind}/{identity}"
    if parameters:
        reference += "?" + "&".join(f"{key}={value}" for key, value in sorted(parameters.items()))
    return TaskEvidence(evidence_kind=kind, reference=reference, content_hash=identity)


def _parameter(reference: str, name: str) -> str:
    values = parse_qs(urlparse(reference).query).get(name, ())
    if len(values) != 1 or len(values[0]) != 64:
        raise ValueError("alternative_evidence.stage_reference_invalid")
    return values[0]


def preparation_intent_hash(
    *,
    request: AlternativeEvidenceRequest,
    obligation: AlternativeEvidenceResearchObligation,
    admission: AlternativeEvidenceAdmission,
    resource_binding_hash: str,
    purpose: str = "PREPARE_PACKET",
) -> str:
    """Hash a preparation intent from Task input without its clock.

    A preparation's Task input carries the request (issuers, cutoff, source
    policy), the obligation, the admission and the resource binding -- and the
    moment the admission was made. Two submissions of the same captured intent
    differ only in that moment, so it is the one field left out here. A new
    cutoff, another issuer scope, another source package, another policy or
    another permission flag is another intent. `purpose` is the Task's own:
    a preparation by default, `FULL_REFRESH` for the single Task that also
    analyses and publishes -- the same request under another purpose is
    another intent.
    """
    return _hash(
        {
            "purpose": purpose,
            "request_hash": request.request_hash,
            "obligation_hash": obligation.obligation_hash,
            "admission": admission.model_dump(
                mode="json", exclude={"admitted_at", "admission_hash"}
            ),
            "resource_binding_hash": resource_binding_hash,
        }
    )


def coverage_unit(
    *,
    unit_id: str,
    ordered_entity_ids: tuple[str, ...],
    request: AlternativeEvidenceRequest,
    obligation: AlternativeEvidenceResearchObligation,
    resource_binding_hash: str,
    network_consent: bool,
    admit_live_official: bool,
    admit_model_review: bool,
) -> AlternativeEvidenceCoverageUnit:
    """Seal one unit of a coverage run under the admission it will be prepared with.

    The unit's preparation intent binds its request, obligation, resource binding and the
    admission's three consent flags; the intent excludes the admission's clock, so the
    request's own cutoff stands in for it (W1: Evidence seals its admission, never its
    caller).

    Args:
        unit_id: The unit's identifier in its run.
        ordered_entity_ids: The unit's issuers, in the order they are read.
        request: The unit's sealed request.
        obligation: The unit's sealed research obligation.
        resource_binding_hash: The source package the unit is prepared from.
        network_consent: Whether the preparation may reach the network.
        admit_live_official: Whether live official sources are admitted.
        admit_model_review: Whether a model review is admitted.

    Returns:
        The sealed unit.
    """
    admission = seal_contract(
        AlternativeEvidenceAdmission,
        "admission_hash",
        request_hash=request.request_hash,
        network_consent=network_consent,
        admit_live_official=admit_live_official,
        admit_model_review=admit_model_review,
        admitted_at=request.evidence_as_of,
    )
    return AlternativeEvidenceCoverageUnit(
        unit_id=unit_id,
        ordered_entity_ids=ordered_entity_ids,
        request=request,
        obligation=obligation,
        preparation_intent_hash=preparation_intent_hash(
            request=request,
            obligation=obligation,
            admission=admission,
            resource_binding_hash=resource_binding_hash,
        ),
    )


def preparation_intent_of(task: TaskRecord) -> str | None:
    """Return a PREPARE_PACKET Task's intent from its durable input."""
    payload = task.input.payload
    if (
        task.task_kind != TASK_KIND
        or task.input.input_schema_id != INPUT_SCHEMA_ID
        or payload.get("purpose") != "PREPARE_PACKET"
    ):
        return None
    return refresh_intent_of(task)


FULL_REFRESH_PURPOSE = "FULL_REFRESH"
"""The intent purpose of the single Task that prepares, analyses and
publishes; its durable input carries no `purpose` key."""


def refresh_intent_of(task: TaskRecord) -> str | None:
    """Return the clockless identity of a single refresh Task.

    Include a preparation or full refresh from its durable input. Return
    ``None`` for a run, submitted analysis, or continuation.
    """
    payload = task.input.payload
    purpose = payload.get("purpose")
    if (
        task.task_kind != TASK_KIND
        or task.input.input_schema_id != INPUT_SCHEMA_ID
        or purpose not in (None, "PREPARE_PACKET")
    ):
        return None
    return preparation_intent_hash(
        request=AlternativeEvidenceRequest.model_validate(payload["request"]),
        obligation=AlternativeEvidenceResearchObligation.model_validate(payload["obligation"]),
        admission=AlternativeEvidenceAdmission.model_validate(payload["admission"]),
        resource_binding_hash=str(payload["resource_binding_hash"]),
        purpose="PREPARE_PACKET" if purpose == "PREPARE_PACKET" else FULL_REFRESH_PURPOSE,
    )


def validate_scoped_obligation(
    *,
    request: AlternativeEvidenceRequest,
    obligation: AlternativeEvidenceResearchObligation,
) -> None:
    """Validate that obligation and request describe one question.

    Three separate agreements, because three different mistakes are possible: a
    scope that named different issuers, a cutoff that lets post-cutoff text in,
    and a source family the request never admitted.
    """
    if obligation.ordered_entity_ids != tuple(request.ordered_entity_ids):
        raise ValueError("alternative_evidence.task_obligation_axis_mismatch")
    if obligation.evidence_as_of != request.evidence_as_of:
        raise ValueError("alternative_evidence.task_obligation_cutoff_mismatch")
    required = {
        SOURCE_FAMILY_BY_EVIDENCE_CLASS[value]
        for value in request.evidence_classes
        if value in SOURCE_FAMILY_BY_EVIDENCE_CLASS
    }
    families = {value.upper() for value in obligation.approved_source_families}
    if not required <= families:
        raise ValueError("alternative_evidence.task_obligation_sources_unadmitted")


def _hash(value: object) -> str:
    return str(canonical_hash(to_jsonable_python(value)))


def _authority_moved(
    resources: AlternativeEvidenceDocumentTaskResources, mode: AlternativeEvidenceMode
) -> str:
    """The refusal of work prepared under authority the Host no longer holds, naming what
    moved (V547): the source mode -- official acquisition served or withdrawn -- else the
    installed package, which an install replaces with every packet prepared under it."""
    live = mode is AlternativeEvidenceMode.LIVE_OFFICIAL
    moved = "source" if live != (resources.live_source is not None) else "package"
    return f"alternative_evidence.task_resource_authority_mismatch:{moved}"


_RUN_STOPPING_CODES = (
    "storage.managed_capacity_exceeded",
    "storage.disk_space_insufficient",
    "alternative_evidence.network_disabled",
)
"""Refusals that name the workspace or its transport rather than one unit:
every later unit would meet the same refusal, so a run stops on the first --
BLOCKED under that code, with the recovery guidance a single preparation
always had -- and a new preparation carries every unit that completed."""


def _stops_the_run(code: str) -> bool:
    return any(code == value or code.endswith(":" + value) for value in _RUN_STOPPING_CODES)


def _failure_code(error: Exception) -> str:
    """The code a stage records for what stopped it, naming the owner's own reason.

    A Workspace knowledge failure (an index that fails its check, a snapshot
    that moved, a semantic pack that is not the pinned one) carries its code
    on ``failure``; a contract refusal raised through pydantic carries its
    code as the first message of a validation report. Both used to be
    recorded as the bare ``task_failed``, which told the person who has to
    repair the workspace nothing about what to repair.
    """
    code = getattr(getattr(error, "failure", None), "code", None)
    if not (isinstance(code, str) and code):
        # A Host owner's refusal (the storage budget's, for one) names its
        # reason on `failure_code`; the Task records that reason, not the
        # bare fact that a stage failed.
        code = getattr(error, "failure_code", None)
    if isinstance(code, str) and code:
        return f"alternative_evidence.task_failed:{code}"[:FAILURE_CODE_MAX_LENGTH]
    value = failure_code_from(error)
    return (
        value if value.startswith("alternative_evidence.") else "alternative_evidence.task_failed"
    )


__all__ = [
    "COVERAGE_RUN_CATEGORY",
    "FULL_REFRESH_PURPOSE",
    "INPUT_SCHEMA_ID",
    "TASK_KIND",
    "UNIT_FAILURE_CATEGORY",
    "UNIT_FAILURE_KIND",
    "AlternativeEvidenceDocumentTaskAdapter",
    "AlternativeEvidenceDocumentTaskResources",
    "EvidenceContinuation",
    "SubmittedEvidenceAnalysis",
    "UnitAuthority",
    "alternative_evidence_document_task_contract",
    "coverage_run_task_contract",
    "coverage_unit",
    "preparation_intent_hash",
    "preparation_intent_of",
    "split_unit_stage",
    "validate_scoped_obligation",
]
