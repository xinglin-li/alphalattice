"""Install local Evidence/CRO authority from a bounded SEC acquisition or captured artifacts.

This is an installation utility, not a product runtime. It reads one verified
historical acquisition, writes a compact recorded-document package inside the
target Research Workspace, and can bind it into an existing workspace manifest.
The Local Web launcher still receives only ``--workspace`` and never reads the
source worktree.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shlex
import sys
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

from alphalattice.control.product_host.composition.application_session import (
    WorkspaceApplicationSession,
)
from alphalattice.control.product_host.composition.evidence_review_workspace import (
    EvidenceReviewArtifactBinding,
    EvidenceReviewModelProfile,
    EvidenceReviewWorkspaceManifest,
    RecordedEvidenceDocumentBundle,
    verify_evidence_review_workspace,
)
from alphalattice.control.product_host.composition.evidence_source_ways import (
    PACKAGE_RULE,
    SETUP_OPTIONS,
)
from alphalattice.control.product_host.composition.portfolio_result_context import (
    installed_authority,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceEvidenceReview,
    read_research_workspace_manifest,
    update_research_workspace_manifest,
)
from alphalattice.control.product_host.composition.retrieval_pack_setup import pack_command
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.input_revisions import (
    ResearchInputRevisions,
)
from alphalattice.control.task_control.contracts import (
    ResearchGoal,
    ResearchPlan,
    StageFailureCause,
    TaskEvidence,
    TaskExecution,
    TaskExecutionCompatibility,
    TaskInputEnvelope,
    TaskLifecycle,
    TaskRecord,
    TaskReplan,
    WorkItemDefinition,
)
from alphalattice.control.task_control.runner import StageDisposition, StageExecutionResult
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.control.workspace_runtime.network_access import network_access
from alphalattice.evidence.alternative_evidence.contracts import (
    DEFAULT_SOURCE_DOCUMENT_BYTES,
    INTEGRATED_FAMILY_SPELLING,
    MATTER_SELECTION_INTEGRATED,
    AlternativeEvidenceClass,
    AlternativeEvidenceMode,
    AlternativeEvidenceRequest,
    AlternativeEvidenceSnapshot,
    AlternativeEvidenceSourcePolicy,
    MatterSelectionPolicy,
    SecIssuerRegistrySnapshot,
    matter_selection_retired,
    seal_contract,
)
from alphalattice.evidence.alternative_evidence.documents.canonicalization import (
    canonicalize_source_documents,
)
from alphalattice.evidence.alternative_evidence.documents.workspace import (
    AlternativeEvidenceDocumentPublisher,
)
from alphalattice.evidence.alternative_evidence.runtime.coverage import index_quiet, sources_short
from alphalattice.evidence.alternative_evidence.runtime.service import (
    AlternativeEvidenceDocumentIntelligenceRuntime,
)
from alphalattice.evidence.alternative_evidence.sources.acquisition import (
    AlternativeEvidenceTaskCancelled,
)
from alphalattice.evidence.alternative_evidence.sources.admission import (
    DEFAULT_SOURCE_CONSENT,
    NOT_GRANTED,
    EvidenceSourceConsent,
    OfficialSourceAdmission,
    admit_official_source,
    seal_live_setup_admission,
    sec_user_agent,
)
from alphalattice.evidence.alternative_evidence.sources.contracts import (
    AcquiredEvidenceDocument,
    AcquiredEvidenceDocumentSet,
    AcquiredEvidenceSourceReferenceSet,
    SecFilingSelectionPlan,
    parse_source_set,
)
from alphalattice.evidence.alternative_evidence.sources.recorded import (
    RecordedEvidenceDocument,
)
from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
    PanelClosureArtifactStore,
)
from alphalattice.foundation.feature_engine.panels.closure_contracts import SectorRevisionMap
from alphalattice.foundation.market_data_ops.storage.duckdb import (
    MarketDataRepository,
)
from alphalattice.interface.local_application.cli_contract import join, refusal_words, shell
from alphalattice.interface.local_application.dispatcher import CommandAdmission
from alphalattice.interface.local_application.failure_codes import setup_failure
from alphalattice.interface.local_application.retrieval_environment import (
    fill_command,
    load,
)
from alphalattice.investment.portfolio_strategy_lab.application.resolution import (
    _risk_surface,
    _sector_map,
)
from alphalattice.investment.risk_research.surfaces.returns import RiskReturnArtifactStore
from alphalattice.kernel.knowledge import model_store
from alphalattice.kernel.knowledge.hybrid import (
    probe_hybrid_retrieval_capabilities,
)
from alphalattice.kernel.knowledge.hybrid_contracts import (
    RECIPE_MINILM_CPU,
    SUPPORTED_RECIPES,
    HybridIndexSpec,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.persistence import replace_with_retry
from alphalattice.kernel.shared_kernel.project_layout import (
    command_prefix,
    resolve_playpen_root,
)
from alphalattice.oversight.chief_risk_officer.portfolio_evidence.contracts import (
    AdmittedListingTicker,
    AdmittedListingTickerAuthority,
    seal_portfolio_evidence_contract,
)

if TYPE_CHECKING:
    from alphalattice.control.workspace_runtime.mutation_gate import WorkspaceMutationGate

PLAYPEN_ROOT = resolve_playpen_root(Path(__file__))


class SourceCoverageRefused(ValueError):
    """A package whose issuers hold too few documents, or none, at its cutoff (V587).

    Its code names the issuers that hold a document, the count the floor needs, those that
    filed nothing and those that failed; the cutoff and the issuers ride beside it, so while
    any failed the refusal offers the same official acquisition again at that one cutoff.
    """

    def __init__(
        self, code: str, *, evidence_as_of: datetime, entities: tuple[str, ...], failed: int
    ) -> None:
        """Name the refusal, the cutoff it was counted at and its issuers.

        Args:
            code: The refusal's code, its numbers after the colon.
            evidence_as_of: The source's cutoff, at which the issuers were counted.
            entities: The package's issuers.
            failed: The issuers counted without a document: a failed or unread acquisition.
        """
        super().__init__(code)
        self.evidence_as_of = evidence_as_of
        self.entities = entities
        self.failed = failed


class OptionRefused(ValueError):
    """A setup refusal on what one option holds, which names that option (V591, V592).

    Its code is the owner's; the option rides beside it, and the setup's answer says what the
    option must hold from the one table its offer states (`SETUP_OPTIONS`).
    """

    def __init__(self, code: str, option: str) -> None:
        """Name the refusal and the option it is on.

        Args:
            code: The refusal's code.
            option: The option, by its name in the answer: a key of `SETUP_OPTIONS`.
        """
        super().__init__(code)
        self.option = option


class SourceFileUnavailable(OptionRefused):
    """A recorded import's file that cannot be read under the option naming its root (V590).

    Its code names the option; the file relative to that root rides beside it, never an
    absolute path, and the OS error is its cause: one the read met, or the object a source set
    names and its knowledge root lacks (V592).
    """

    def __init__(self, option: str, root: Path, error: OSError) -> None:
        """Name the option and the file under its root that could not be read.

        Args:
            option: The root option the file lies under, a key of `SETUP_OPTIONS`.
            root: That option's directory, as resolved.
            error: The OS error the read met.
        """
        super().__init__(f"evidence_review.source_file_unavailable:{option}", option)
        named = Path(error.filename).resolve() if isinstance(error.filename, str) else None
        self.file = (
            named.relative_to(root.resolve()).as_posix()
            if named is not None and named.is_relative_to(root.resolve())
            else None
        )


def _read(path: Path, model: Any):  # type: ignore[no-untyped-def]
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _read_source(root: Path, category: str, identity: str, model: type, field: str):  # type: ignore[no-untyped-def]
    path = (root / category / f"{identity}.json").resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("evidence_review.source_path_escapes_root")
    if model is AcquiredEvidenceDocumentSet:
        # Either durable form of a source set: the retained inline sets and
        # the reference sets whose bytes the source's evidence Workspace holds.
        value = parse_source_set(json.loads(path.read_text(encoding="utf-8")))
    else:
        value = _read(path, model)
    if getattr(value, field) != identity:
        raise ValueError("evidence_review.source_identity_mismatch")
    return value


@contextmanager
def _reading(option: str | None, root: Path) -> Iterator[None]:
    """Refuse a file a recorded import cannot read under `option`'s root by that option (V590).

    An acquisition's own reads, under the workspace it has just written, keep the setup's file
    refusal (`option` None).
    """
    try:
        yield
    except OSError as error:
        if option is None:
            raise
        raise SourceFileUnavailable(option, root, error) from error


def _holding_root(given: Path, source_set_hash: str) -> Path | None:
    """The one directory below `given` that holds the named source set, where one does (V590).

    A root named one level or more above the artifact store -- the workspace, its runtime or
    its artifacts -- is the mistake a recorded import meets; the set's own file decides.
    """
    holding = [
        candidate
        for candidate in (
            given / "alternative-evidence",
            given / "artifacts" / "alternative-evidence",
            given / "runtime" / "artifacts" / "alternative-evidence",
        )
        if (candidate / "source-document-sets" / f"{source_set_hash}.json").is_file()
    ]
    return holding[0] if len(holding) == 1 else None


def _sealed_plans(root: Path, entities: Collection[str]) -> dict[str, list[SecFilingSelectionPlan]]:
    """The source's sealed selection plans of these issuers: what its acquisitions read of each.

    A plan that does not read back under its own identity establishes nothing, so it is passed
    over and its issuer counts against the floor.
    """
    plans: dict[str, list[SecFilingSelectionPlan]] = {}
    directory = root / "sec-selection-plans"
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else ():
        try:
            plan = _read_source(
                root, "sec-selection-plans", path.stem, SecFilingSelectionPlan, "plan_hash"
            )
        except (OSError, ValueError):
            continue
        if plan.entity_id in entities:
            plans.setdefault(plan.entity_id, []).append(plan)
    return plans


def _source_documents(
    source_root: Path,
    source_set: AcquiredEvidenceDocumentSet | AcquiredEvidenceSourceReferenceSet,
    *,
    knowledge_root: Path | None = None,
    option: str | None = None,
) -> tuple[AcquiredEvidenceDocument, ...]:
    """The set's documents with their bytes, resolved from the source's own store.

    A recorded import names its knowledge root (`option`): a root that is not there, or an object
    the set names that the root lacks, is refused by that option with the object's place under
    it (V592), before the store's owner is opened there and creates its folders.
    """
    if isinstance(source_set, AcquiredEvidenceDocumentSet):
        return tuple(source_set.documents)
    if knowledge_root is None:
        knowledge_root = source_root.parent.parent / "evidence-knowledge"
    missing = (
        next(
            (
                knowledge_root / value.content_object_path
                for value in source_set.documents
                if not (knowledge_root / value.content_object_path).is_file()
            ),
            None,
        )
        if knowledge_root.is_dir()
        else knowledge_root
    )
    if missing is not None and option is not None:
        lacking = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(missing))
        raise SourceFileUnavailable(option, knowledge_root, lacking) from lacking
    if missing == knowledge_root:
        raise ValueError("evidence_review.source_objects_root_missing")
    publisher = AlternativeEvidenceDocumentPublisher(knowledge_root)
    return tuple(publisher.resolve_source_object(value) for value in source_set.documents)


def _publish(
    path: Path, value: object, identity: str, *, workspace: Path
) -> EvidenceReviewArtifactBinding:
    if not path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("evidence_review.package_path_escapes_workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump_json(indent=2).encode("utf-8") + b"\n"  # type: ignore[attr-defined]
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("evidence_review.immutable_package_conflict")
        return EvidenceReviewArtifactBinding(
            relative_path=path.relative_to(workspace).as_posix(),
            file_sha256=hashlib.sha256(payload).hexdigest(),
            content_hash=identity,
        )
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return EvidenceReviewArtifactBinding(
        relative_path=path.relative_to(workspace).as_posix(),
        file_sha256=hashlib.sha256(payload).hexdigest(),
        content_hash=identity,
    )


def materialize(arguments: argparse.Namespace) -> dict[str, object]:
    """Read the captured source document under its admitted contract."""
    _check(arguments)
    with WorkspaceApplicationSession.acquire(arguments.workspace.resolve()) as session:
        return run_setup(arguments, session)


def _check(arguments: argparse.Namespace) -> None:
    if bool(arguments.model_name) != bool(arguments.deepseek_base_url):
        raise OptionRefused(
            "evidence_review.managed_profile_requires_model_and_endpoint", "model_name"
        )
    floor = arguments.minimum_entity_coverage
    if floor is not None and (not math.isfinite(floor) or not 0 <= floor <= 1):
        raise OptionRefused(
            "evidence_review.minimum_entity_coverage_invalid", "minimum_entity_coverage"
        )
    live = bool(getattr(arguments, "acquire_sec", False))
    rebind = bool(getattr(arguments, "rebind_installed", False))
    archived = arguments.source_artifact_root is not None and arguments.source_set_hash is not None
    if rebind and (live or arguments.source_artifact_root or arguments.source_set_hash):
        raise ValueError("evidence_review.rebind_takes_no_source")
    if not rebind and (
        live == archived or (live and (arguments.source_artifact_root or arguments.source_set_hash))
    ):
        raise ValueError("evidence_review.choose_sec_acquisition_or_source_artifacts")
    if archived and not re.fullmatch(r"[0-9a-f]{64}", arguments.source_set_hash or ""):
        raise OptionRefused("evidence_review.source_set_hash_invalid", "source_set_hash")
    if rebind and getattr(arguments, "preflight", False):
        raise ValueError("evidence_review.preflight_requires_a_source")


def setup_arguments(argv: Sequence[str], workspace: Path) -> argparse.Namespace:
    """The setup's command line as the running Host takes it, on the Host's own workspace."""
    arguments = _parser().parse_args([*argv, "--workspace", str(workspace)])
    _check(arguments)
    return arguments


_CARRIED_REFUSAL = frozenset({"failure_code", "location", "next_commands"})
"""What a blocked install keeps of its refusal beside the Task's code: its subject, its option and
its way on."""
EVIDENCE_INSTALL_TASK_KIND = "evidence_install"
INSTALL_STAGES = ("environment", "model", "acquisition", "index", "publish")
INSTALL_OUTPUTS = Path("runtime") / "evidence-install"
"""Each install Task's stage outputs, by Task id: what a later stage, or a recovery, reads."""


class EvidenceInstall:
    """An Evidence package installed in the running Host, as a Task of the setup's steps.

    Each stage keeps its outputs beside the Task, so a recovery runs only the stage it stopped
    in; a cancel stops the acquisition between requests; once bound, the Host serves the
    package (`installed`) with no restart.
    """

    task_kind = EVIDENCE_INSTALL_TASK_KIND
    replans = (TaskReplan(task_kind=EVIDENCE_INSTALL_TASK_KIND, admitting="EVIDENCE_INSTALL"),)

    def __init__(
        self,
        *,
        session: WorkspaceApplicationSession,
        clock: Callable[[], datetime],
        installed: Callable[[], None],
        official_source: OfficialSourceAdmission | None = None,
    ) -> None:
        """Bind the install to the Host's session, its clock and how it serves a new package."""
        self.session, self.clock, self.installed = session, clock, installed
        self.official_source = official_source

    def admit(self, argv: Sequence[str]) -> CommandAdmission:
        """Admit one install of the setup's command line."""
        kind = EVIDENCE_INSTALL_TASK_KIND
        envelope = TaskInputEnvelope.create(
            task_kind=EVIDENCE_INSTALL_TASK_KIND,
            input_schema_id="evidence-install",
            payload={"arguments": list(argv), "requested_at": self.clock().isoformat()},
        )
        goal = ResearchGoal.create(
            goal_kind="INSTALL_EVIDENCE_PACKAGE",
            input_hash=envelope.input_hash,
            deliverable_kind="EvidenceReviewWorkspaceManifest",
            summary="Install an Evidence source package in the running Host and serve it.",
        )
        plan = ResearchPlan.create(
            goal_hash=goal.goal_hash,
            workflow_definition_hash=canonical_hash(INSTALL_STAGES),
            verifier_catalog_hash=canonical_hash([f"{kind}.{stage}" for stage in INSTALL_STAGES]),
            work_items=tuple(
                WorkItemDefinition.create(
                    stage_id=stage,
                    dependency_ids=INSTALL_STAGES[:position],
                    verifier_id=f"{kind}.{stage}",
                )
                for position, stage in enumerate(INSTALL_STAGES)
            ),
        )
        record = self.session.task_control_registry.admit(
            input_envelope=envelope, goal=goal, plan=plan, observed_at=self.clock()
        ).record
        return CommandAdmission(task_id=record.task_id, lifecycle=record.lifecycle.value)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Run or recover an admitted install through Task Control's runner."""
        task = self.session.task_control_registry.task(task_id)
        self.session.execute_admitted(task, self, self.clock, expected_task_hash)

    def compatibility(self, task: TaskRecord) -> TaskExecutionCompatibility:
        """An install resumes under the stages it was admitted with."""
        stages = canonical_hash(INSTALL_STAGES)
        return TaskExecutionCompatibility.create(
            task_contract_hash=canonical_hash({EVIDENCE_INSTALL_TASK_KIND: 1}),
            workflow_definition_hash=task.plan.workflow_definition_hash,
            input_schema_id=task.input.input_schema_id,
            domain_policy_hash=stages,
            framework_identity_hash=self.session.execution_identity(stages),
        )

    def execute_stage(
        self, *, task: TaskRecord, execution: TaskExecution, work_item: WorkItemDefinition
    ) -> StageExecutionResult:
        """Run one stage of the install.

        A refusal blocks the Task by its code and keeps its way on; a cancel stops it.
        """
        del execution
        stage, path = work_item.stage_id, self._path(task)
        held = self._outputs(task)
        arguments = setup_arguments(task.input.payload["arguments"], self.session.workspace)
        try:
            if stage == "environment" and not load():
                raise ValueError("evidence_review.retrieval_environment_not_loaded")
            answer = _step(
                _Setup(
                    arguments, self.session, official_source=self.official_source, clock=self.clock
                ),
                stage,
                held,
                cancelled=lambda: (
                    self.session.task_control_registry.task(task.task_id).lifecycle
                    is TaskLifecycle.CANCEL_REQUESTED
                ),
            )
            held[stage] = (
                {key: answer.get(key) for key in ("authority_hash", "package_id", "installed")}
                if stage == INSTALL_STAGES[-1]
                else answer
            )
            if held[stage].get("installed"):
                self.installed()
        except AlternativeEvidenceTaskCancelled:
            return StageExecutionResult(StageDisposition.CANCELLED)
        except (ValueError, RuntimeError, OSError) as error:
            refusal = setup_refusal(arguments, error)
            code = str(refusal["failure_code"])
            cause = StageFailureCause(
                exception_type=type(error).__name__,
                detail=json.dumps({k: v for k, v in refusal.items() if k in _CARRIED_REFUSAL})[
                    :400
                ],
                step=stage,
            )
            return StageExecutionResult(
                StageDisposition.BLOCKED,
                failure_code=code if len(code) <= 120 else code.split(":")[0],
                failure_cause=cause,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f"{path.name}.partial")
        staged.write_bytes(json.dumps(held, sort_keys=True).encode("utf-8"))
        replace_with_retry(staged, path)
        reference = f"playpen://evidence-install/{task.task_id}/{stage}"
        evidence = TaskEvidence(
            evidence_kind=f"{EVIDENCE_INSTALL_TASK_KIND}.{stage}",
            reference=reference,
            content_hash=canonical_hash(held[stage]),
        )
        return StageExecutionResult(StageDisposition.READY, evidence=(evidence,))

    def verify_stage(
        self,
        *,
        task: TaskRecord,
        execution: TaskExecution,
        work_item: WorkItemDefinition,
        evidence: tuple[TaskEvidence, ...],
    ) -> tuple[TaskEvidence, ...]:
        """The stage's kept outputs still hash to its evidence."""
        del execution
        held = self._outputs(task).get(work_item.stage_id)
        if held is None or [item.content_hash for item in evidence] != [canonical_hash(held)]:
            raise ValueError("evidence_review.install_failed")
        return evidence

    def _path(self, task: TaskRecord) -> Path:
        return self.session.workspace / INSTALL_OUTPUTS / f"{task.task_id}.json"

    def _outputs(self, task: TaskRecord) -> dict[str, Any]:
        try:
            return cast(dict[str, Any], json.loads(self._path(task).read_bytes()))
        except FileNotFoundError:
            return {}


class EvidenceInstallCommand:
    """One install, as the Host's dispatcher runs it; a recovery rebuilds it from its Task."""

    command_kind = EVIDENCE_INSTALL_TASK_KIND

    def __init__(self, install: EvidenceInstall, argv: Sequence[str] = ()) -> None:
        """One install of the setup's command line."""
        self.install, self.argv = install, tuple(argv)

    def admit(self) -> CommandAdmission:
        """Admit the install's Task."""
        return self.install.admit(self.argv)

    def execute(self, task_id: UUID, *, expected_task_hash: str | None = None) -> None:
        """Run or recover the install's Task."""
        self.install.execute(task_id, expected_task_hash=expected_task_hash)


def _listing_authority(
    universe: dict[str, str], registry: SecIssuerRegistrySnapshot, entities: tuple[str, ...]
) -> AdmittedListingTickerAuthority:
    """Every universe listing's own symbol, admitted; identity stays the registry's.

    A holding's SEC identity comes from the captured official registry, never
    from a symbol alone. Admitting the universe's symbols for every listing
    lets the review type each holding it cannot read: a symbol the registry
    does not carry is `TICKER_NOT_IN_SEC_REGISTRY`, an identified issuer
    without a retained filing is a missing source -- rather than leaving the
    whole book outside the authority when only a few issuers were acquired.
    The acquisition scope bounds documents, not identity.
    """
    registry_by_entity = {value.entity_id: value for value in registry.entries}
    if set(entities) - set(registry_by_entity):
        raise RuntimeError("evidence_review.registry_missing_materialized_entity")
    missing = tuple(
        sorted(
            registry_by_entity[entity].ticker
            for entity in entities
            if registry_by_entity[entity].ticker not in universe
        )
    )
    if missing:
        raise RuntimeError("evidence_review.universe_missing_admitted_ticker:" + ",".join(missing))
    return seal_portfolio_evidence_contract(
        AdmittedListingTickerAuthority,
        "authority_hash",
        # Ordered by listing id, which is the key the authority is joined on
        # and the order its own contract requires.
        entries=tuple(
            sorted(
                (
                    AdmittedListingTicker(listing_id=listing_id, ticker=symbol.strip().upper())
                    for symbol, listing_id in universe.items()
                ),
                key=lambda value: value.listing_id,
            )
        ),
    )


class _Setup:
    """One setup's steps on a session its caller holds.

    The script runs them at once (`run_setup`), the Host's install Task one stage each
    (`EvidenceInstall`).
    """

    def __init__(
        self,
        arguments: argparse.Namespace,
        session: WorkspaceApplicationSession,
        *,
        official_source: OfficialSourceAdmission | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Read the workspace, its research input, universe and recipe, and the model profile."""
        workspace = arguments.workspace.resolve()
        input_id = getattr(arguments, "research_input_id", None)
        input_hash = getattr(arguments, "research_input_hash", None)
        if (input_id is None) != (input_hash is None):
            raise OptionRefused(
                "evidence_review.research_input_id_and_hash_required_together", "research_input_id"
            )
        if input_id is not None:
            selected = ResearchInputRevisions(session).select(input_id, input_hash)
            universe = _research_universe_listing_ids(workspace, selected.binding_hash)
        else:
            if read_research_workspace_manifest(workspace).strategy_installation == "NOT_INSTALLED":
                raise OptionRefused(
                    "evidence_review.research_input_selection_required", "research_input_id"
                )
            universe = _universe_listing_ids(workspace)
        recipe = arguments.recipe or RECIPE_MINILM_CPU
        if recipe not in SUPPORTED_RECIPES:
            raise ValueError("evidence_review.retrieval_recipe_unsupported")

        self.arguments, self.session, self.workspace = arguments, session, workspace
        self.official_source = official_source
        self.clock = clock
        self.universe, self.recipe, self.research_input = universe, recipe, (input_id, input_hash)
        self.profile = (
            None
            if not arguments.model_name
            else EvidenceReviewModelProfile.create(
                model_name=arguments.model_name,
                base_url=arguments.deepseek_base_url,
                analyst_timeout_seconds=arguments.analyst_timeout_seconds,
                review_timeout_seconds=arguments.review_timeout_seconds,
            )
        )

    def capability(self) -> tuple[str, str]:
        """The recipe's packs bound into the workspace, and their capability: its root, its hash.

        A retrieval runtime filled after this process started is read first (`load`).
        """
        load()
        arguments, workspace, recipe = self.arguments, self.workspace, self.recipe
        semantic_root = _semantic_root(arguments, workspace=workspace, recipe=recipe)
        try:
            semantic_relative = semantic_root.relative_to(workspace).as_posix()
        except ValueError as error:
            raise OptionRefused(
                "evidence_review.semantic_model_outside_workspace", "semantic_model"
            ) from error
        capability = probe_hybrid_retrieval_capabilities(
            semantic_root, HybridIndexSpec.for_recipe(recipe)
        )
        if capability.status != "READY":
            raise RuntimeError("evidence_review.semantic_capability_not_ready")
        return semantic_relative, str(capability.logical_hash)

    def sources(
        self, capability_hash: str, cancelled: Callable[[], bool] | None = None
    ) -> tuple[Path, str, dict[str, object]] | dict[str, object]:
        """The source set to package, acquired under consent or a recorded import's.

        An acquisition's preflight answers instead.
        """
        arguments, workspace, universe = self.arguments, self.workspace, self.universe
        if getattr(arguments, "acquire_sec", False):
            requested = tuple(sorted(value.strip().upper() for value in arguments.entities or ()))
            if not requested or len(requested) > 8 or len(set(requested)) != len(requested):
                raise OptionRefused("evidence_review.entity_scope_invalid", "entities")
            if any(value not in universe for value in requested):
                raise OptionRefused("evidence_review.entity_scope_outside_universe", "entities")
            maximum_documents = arguments.maximum_documents_per_issuer
            if not 3 <= maximum_documents <= 20 or len(requested) * maximum_documents > 24:
                raise OptionRefused(
                    "evidence_review.document_budget_exceeds_retrieval_capacity",
                    "maximum_documents_per_issuer",
                )
            consent = bool(arguments.network_consent)
            network_enabled = network_access(workspace).allowed
            user_agent = sec_user_agent()
            cutoff = _acquisition_cutoff(arguments, self.clock())
            scopes = _accession_scopes(arguments, requested)
            if arguments.preflight:
                return {
                    "status": "EVIDENCE_SOURCE_PREFLIGHT",
                    "entity_ids": requested,
                    "maximum_documents": len(requested) * maximum_documents,
                    "evidence_as_of": None if cutoff is None else cutoff.isoformat(),
                    "accession_scopes": {k: sorted(v) for k, v in scopes.items()},
                    "maximum_document_bytes": arguments.maximum_document_bytes,
                    "semantic_capability_hash": capability_hash,
                    "network_consent": consent,
                    "network_enabled": network_enabled,
                    "sec_contact_configured": user_agent is not None,
                    "managed_model_required": False,
                    "acquisition_performed": False,
                    "claim": "Preflight grants no acquisition, coverage or historical "
                    "availability.",
                }
            if not consent:
                raise ValueError("evidence_review.explicit_sec_network_consent_required")
            source_root, source_hash, acquisition = _acquire_sec_sources(
                arguments,
                requested,
                official_source=self.official_source,
                clock=self.clock,
                cutoff=cutoff,
                accession_scopes=scopes,
                cancelled=cancelled,
            )
        else:
            requested = ()
            source_root, source_hash = (
                arguments.source_artifact_root.resolve(),
                arguments.source_set_hash,
            )
            acquisition = {"mode": "RECORDED_IMPORT", "network_calls": 0}
        return source_root, source_hash, acquisition

    def package(
        self,
        capability_hash: str,
        source_root: Path,
        source_hash: str,
        acquisition: dict[str, object],
    ) -> dict[str, Any]:
        """The source set read and judged against its floor, as the seal takes it.

        A recorded import's preflight answers instead (its `status`).
        """
        arguments, universe = self.arguments, self.universe
        input_id, input_hash = self.research_input
        requested: tuple[str, ...] = ()
        # A recorded import names its roots, and a file it cannot read under one is refused naming
        # the option, the file under it and what the option should hold (V590).
        imported = acquisition["mode"] == "RECORDED_IMPORT"
        with _reading("source_artifact_root" if imported else None, source_root):
            source_set = _read_source(
                source_root,
                "source-document-sets",
                source_hash,
                AcquiredEvidenceDocumentSet,
                "source_set_hash",
            )
            snapshot = _read_source(
                source_root,
                "snapshots",
                source_set.source_snapshot_hash,
                AlternativeEvidenceSnapshot,
                "snapshot_hash",
            )
            registry = _read_source(
                source_root,
                "registries",
                snapshot.registry_hash,
                SecIssuerRegistrySnapshot,
                "registry_hash",
            )
            if snapshot.request_hash != source_set.request_hash:
                raise ValueError("evidence_review.source_request_mismatch")
            source_request = _read_source(
                source_root,
                "requests",
                source_set.request_hash,
                AlternativeEvidenceRequest,
                "request_hash",
            )
        if not requested:
            requested = tuple(
                sorted(
                    value.strip().upper()
                    for value in arguments.entities or source_request.ordered_entity_ids
                )
            )
        if (
            not requested
            or len(set(requested)) != len(requested)
            or not set(requested) <= set(source_request.ordered_entity_ids)
        ):
            raise OptionRefused("evidence_review.entity_scope_not_in_source_request", "entities")
        knowledge_root = (
            arguments.source_knowledge_root.resolve()
            if getattr(arguments, "source_knowledge_root", None) is not None
            else None
        )
        objects = "source_knowledge_root" if imported else None
        with _reading(objects, knowledge_root or source_root.parent.parent / "evidence-knowledge"):
            documents = _source_documents(
                source_root, source_set, knowledge_root=knowledge_root, option=objects
            )
        admitted, rejected = canonicalize_source_documents(
            tuple(value for value in documents if value.entity_id in requested)
        )
        # The floor as the coverage run judges it (V541, V587): an issuer whose filing index at
        # the source's cutoff shows nothing filed in the window holds nothing to count and leaves
        # the share; a failed or unread acquisition stays in it. Said whether it installs or not.
        covered = {value.entity_id for value in admitted}
        uncovered = tuple(entity for entity in requested if entity not in covered)
        filed_nothing = (
            index_quiet(
                _sealed_plans(source_root, uncovered),
                evidence_as_of=source_request.evidence_as_of,
                window_days=source_request.source_policy.sec_recent_8k_days,
            )
            if uncovered
            else frozenset()
        )
        failed = tuple(entity for entity in uncovered if entity not in filed_nothing)
        if not admitted:
            raise SourceCoverageRefused(
                "evidence_review.materialization_has_no_admitted_documents:"
                f"{len(filed_nothing)} of {len(requested)} issuers filed nothing in the window, "
                f"{len(failed)} failed",
                evidence_as_of=source_request.evidence_as_of,
                entities=requested,
                failed=len(failed),
            )
        captured = tuple(
            RecordedEvidenceDocument(
                entity_id=value.entity_id,
                source_right="USER_PROVIDED_FOR_LOCAL_RESEARCH",
                evidence_class=AlternativeEvidenceClass.ISSUER_OFFICIAL_RECORDED,
                document_type=value.document_type,
                revision=value.revision,
                published_at=value.published_at,
                captured_at=source_set.acquired_at
                if value.source_name == "SEC_EDGAR"
                else value.available_at,
                # A recorded import is available no earlier than its capture; the
                # original's acceptance, date precision and report period travel
                # beside that bound rather than being lost to it.
                available_at=max(value.available_at, source_set.acquired_at)
                if value.source_name == "SEC_EDGAR"
                else value.available_at,
                accepted_at=value.accepted_at,
                published_precision=value.published_precision,
                report_period_end=value.report_period_end,
                text=value.canonical_markdown.decode("utf-8"),
                immutable_source=True,
                # Ordered and deduplicated: re-materializing an already
                # materialized source set carries these two lines in
                # `value.limitations`, and appending them again rewrote the bundle
                # hash over byte-identical text.
                limitations=tuple(
                    dict.fromkeys(
                        (
                            "Captured SEC EDGAR filing materialized for offline local research."
                            if value.source_name == "SEC_EDGAR"
                            else "Admitted recorded-local source; no new SEC acquisition.",
                            f"Canonical source content hash: {value.canonical_content_hash}.",
                            *value.limitations,
                        )
                    )
                ),
            )
            for value in admitted
        )
        bundle = RecordedEvidenceDocumentBundle.create(captured)
        entities = requested
        floor = (
            0.60 if arguments.minimum_entity_coverage is None else arguments.minimum_entity_coverage
        )
        short = sources_short(
            entities, covered, floor=floor, quiet=lambda _uncovered: filed_nothing
        )
        if short is not None:
            raise SourceCoverageRefused(
                "evidence_review.materialization_issuer_coverage_incomplete:"
                f"{short.held} of {short.issuers} counted issuers hold a document, "
                f"{short.needed} needed, {short.quiet} filed nothing, "
                f"{len(short.uncovered)} failed",
                evidence_as_of=source_request.evidence_as_of,
                entities=entities,
                failed=len(short.uncovered),
            )
        listing = _listing_authority(universe, registry, entities)
        standing = {
            "hold_a_document": sorted(covered),
            "filed_nothing": sorted(filed_nothing),
            "failed": list(failed),
        }
        if getattr(arguments, "preflight", False):
            # The recorded import's own check (V590): its records, scope, documents and floor read
            # and judged as the install reads them, with no package sealed and no binding moved.
            return {
                "status": "EVIDENCE_SOURCE_PREFLIGHT",
                "source": "RECORDED_IMPORT",
                "entity_ids": entities,
                "source_set_hash": source_set.source_set_hash,
                "evidence_as_of": source_request.evidence_as_of.isoformat(),
                "admitted_document_count": len(bundle.documents),
                "rejected_document_count": len(rejected),
                "issuer_coverage": standing,
                "minimum_entity_coverage": floor,
                "semantic_capability_hash": capability_hash,
                "installed": False,
                "claim": "Preflight seals no package and moves no binding; installing reads "
                "the same records.",
            }
        return {
            "registry": registry,
            "listing": listing,
            "bundle": bundle,
            "entities": entities,
            "minimum_entity_coverage": floor,
            "facts": {
                "source_set_hash": source_set.source_set_hash,
                "source_artifact_root": str(source_root),
                "acquisition": acquisition,
                "research_input_id": input_id,
                "research_input_hash": input_hash,
                "source_document_count": len(source_set.documents),
                "rejected_document_count": len(rejected),
                "issuer_coverage": standing,
            },
        }

    def install(
        self, semantic_relative: str, capability_hash: str, package: dict[str, Any] | None
    ) -> dict[str, object]:
        """Seal and bind the package, or with none rebind the installed one."""
        if package is None:
            return _rebind_installed(
                self.arguments,
                workspace=self.workspace,
                gate=self.session.mutation_gate,
                universe=self.universe,
                semantic_relative=semantic_relative,
                capability_hash=capability_hash,
                profile=self.profile,
                research_input=self.research_input,
            )
        return _seal_package(
            self.arguments,
            workspace=self.workspace,
            gate=self.session.mutation_gate,
            semantic_relative=semantic_relative,
            capability_hash=capability_hash,
            profile=self.profile,
            authority_id=self.arguments.authority_id or DEFAULT_AUTHORITY_ID,
            selection=_matter_selection(self.arguments),
            **package,
        )


def run_setup(
    arguments: argparse.Namespace,
    session: WorkspaceApplicationSession,
    *,
    official_source: OfficialSourceAdmission | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, object]:
    """Run the setup on a session the caller holds: its preflight answer or its installation."""
    setup, held = (
        _Setup(arguments, session, official_source=official_source, clock=clock),
        dict[str, Any](),
    )
    for stage in INSTALL_STAGES:
        held[stage] = _step(setup, stage, held)
        if held[stage].get("status") == "EVIDENCE_SOURCE_PREFLIGHT":
            break
    return cast(dict[str, object], held[stage])


def _step(
    setup: _Setup,
    stage: str,
    held: dict[str, Any],
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """One stage of the setup over the outputs of the stages before it.

    A preflight answers in place of the acquisition or the index (its `status`); the
    publication is the answer.
    """
    rebind = bool(getattr(setup.arguments, "rebind_installed", False))
    if stage == "environment":
        return {"recipe": setup.recipe}
    if stage == "model":
        relative, capability = setup.capability()
        return {"semantic_relative": relative, "capability_hash": capability}
    model = held["model"]
    if stage == "acquisition":
        if rebind:
            return {}
        sources = setup.sources(model["capability_hash"], cancelled)
        if isinstance(sources, dict):
            return sources
        root, source_hash, acquisition = sources
        return {"source_root": str(root), "source_hash": source_hash, "acquisition": acquisition}
    found = held["acquisition"]
    package = (
        None
        if rebind
        else setup.package(
            model["capability_hash"],
            Path(found["source_root"]),
            found["source_hash"],
            found["acquisition"],
        )
    )
    if package is not None and "status" in package:
        return package
    bundle = None if package is None else package["bundle"].bundle_hash
    if stage == "index":
        return {"bundle_hash": bundle}
    # The package is computed again from the kept source set; the index stage's hash binds it.
    if bundle != held["index"]["bundle_hash"]:
        raise ValueError("evidence_review.install_failed")
    if cancelled():
        raise AlternativeEvidenceTaskCancelled("alternative_evidence.current_task_cancelled")
    return setup.install(model["semantic_relative"], model["capability_hash"], package)


_DELEGATED = frozenset(
    {
        *("workspace", "acquire_sec", "entities", "accessions", "evidence_as_of"),
        *("maximum_documents_per_issuer", "maximum_document_bytes", "network_consent"),
        *("install", "preflight", "research_input_id", "research_input_hash", "semantic_model"),
    }
)
"""The options a first use's agent may install with: its acquisition's own choices."""


def within_delegation(arguments: argparse.Namespace, current: EvidenceSourceConsent) -> bool:
    """Whether a first use's agent may install with these options.

    Only the acquisition's own choices, within the default budget and the person's own consent.
    """
    defaults = vars(_parser().parse_args(["--workspace", str(arguments.workspace)]))
    if any(vars(arguments)[name] != defaults[name] for name in defaults.keys() - _DELEGATED):
        return False
    per_issuer = arguments.maximum_documents_per_issuer
    documents = per_issuer * max(len(arguments.entities or ()), 1)
    wanted = EvidenceSourceConsent(
        documents_per_issuer=per_issuer,
        total_documents=documents,
        total_bytes=documents * (arguments.maximum_document_bytes or DEFAULT_SOURCE_DOCUMENT_BYTES),
    )
    return wanted.within(DEFAULT_SOURCE_CONSENT) and (
        current.actor in NOT_GRANTED
        or current.actor.startswith("first-use-goal:")
        or wanted.within(current)
    )


def _semantic_root(arguments: argparse.Namespace, *, workspace: Path, recipe: str) -> Path:
    """Where the recipe's packs are read from, inside the workspace.

    Every recipe is bound from the application model store: a small directory
    of links pointing at the store's verified packs, so one physical copy serves
    every workspace and a pack not installed is a refusal naming it. The retained
    recipe keeps its own layout under `evidence-cro-authority/semantic-model`
    (the encoder at the root, V208), or an explicit `--semantic-model` directory;
    a later recipe's is `semantic-model-<recipe>`. Packs are installed with
    scripts/install_retrieval_pack.py, never here.
    """
    store = (
        Path(arguments.model_store).resolve()
        if arguments.model_store
        else model_store.default_store_root()
    )
    if recipe == RECIPE_MINILM_CPU:
        if arguments.semantic_model is not None:
            return Path(arguments.semantic_model).resolve()
        try:
            return cast(
                Path,
                model_store.bind_retained_recipe(
                    store, workspace / "evidence-cro-authority" / "semantic-model"
                ),
            )
        except model_store.ModelStoreError as error:
            raise ValueError(str(error)) from error
    try:
        return Path(
            model_store.bind_recipe(
                store, recipe, workspace / "evidence-cro-authority" / f"semantic-model-{recipe}"
            )
        )
    except model_store.ModelStoreError as error:
        raise ValueError(str(error)) from error


def _rebind_installed(
    arguments: argparse.Namespace,
    *,
    workspace: Path,
    gate: WorkspaceMutationGate,
    universe: dict[str, str],
    semantic_relative: str,
    capability_hash: str,
    profile: EvidenceReviewModelProfile | None,
    research_input: tuple[str | None, str | None],
) -> dict[str, object]:
    """Re-seal the installed package's listing authority over the universe.

    No source, no acquisition, no network: the registry, the recorded
    documents, the semantic pack and (unless re-declared) the model profile,
    authority id, coverage floor and matter selection are the installed ones,
    verified before anything is written. What changes is which listings the authority admits
    a symbol for -- an install made when the authority named only the
    acquired issuers is widened to its universe without a new capture.
    """
    binding = read_research_workspace_manifest(workspace).evidence_review
    if binding is None:
        raise ValueError("evidence_review.rebind_requires_installed_authority")
    # The pack was probed once above; the installed manifest must declare
    # that same capability, which is what the verifier checks with it -- unless
    # a recipe is named, when the installed authority is verified with its
    # own packs (a real probe) before its successor is sealed.
    verified = verify_evidence_review_workspace(
        workspace=workspace,
        binding=binding,
        semantic_capability_reader=(
            None if arguments.recipe else (lambda _runtime: capability_hash)
        ),
    )
    verified.runtime.close()
    installed = verified.manifest
    if getattr(arguments, "matter_selection", None) is None and matter_selection_retired(
        installed.matter_selection
    ):
        # Kept like the id, floor and profile -- unless it is retired: a
        # rebind never re-seals a retired selection, and never moves the
        # workspace to the integrated one unasked.
        raise ValueError(
            "evidence_review.installed_matter_selection_retired:"
            "rebind with --matter-selection INTEGRATED_TOPIC_ROUTING"
        )
    recipe = arguments.recipe or RECIPE_MINILM_CPU
    if installed.semantic_model_relative_path != semantic_relative and (
        recipe == installed.retrieval_recipe
    ):
        # The pack under an installed authority is never swapped silently; an
        # explicit `--recipe` names a different recipe and its own packs.
        raise OptionRefused("evidence_review.rebind_semantic_model_mismatch", "semantic_model")
    entities = tuple(sorted({value.entity_id for value in verified.documents.documents}))
    listing = _listing_authority(universe, verified.registry, entities)
    return _seal_package(
        arguments,
        workspace=workspace,
        gate=gate,
        registry=verified.registry,
        listing=listing,
        bundle=verified.documents,
        entities=entities,
        semantic_relative=semantic_relative,
        capability_hash=capability_hash,
        profile=installed.model_profile if arguments.model_name is None else profile,
        authority_id=arguments.authority_id or installed.authority_id,
        minimum_entity_coverage=(
            installed.minimum_entity_coverage
            if arguments.minimum_entity_coverage is None
            else arguments.minimum_entity_coverage
        ),
        # Kept like the id, floor and profile: a rebind that names no selection
        # never moves the workspace to another one, whatever a fresh install's
        # default is.
        selection=(
            installed.matter_selection
            if getattr(arguments, "matter_selection", None) is None
            else _matter_selection(arguments)
        ),
        facts={
            "source_set_hash": None,
            "source_artifact_root": None,
            "acquisition": {"mode": "REBIND_INSTALLED", "network_calls": 0},
            "research_input_id": research_input[0],
            "research_input_hash": research_input[1],
            "source_document_count": len(verified.documents.documents),
            "rejected_document_count": 0,
            "rebound_from_authority_hash": installed.authority_hash,
            "listing_entries_before": len(verified.listing_authority.entries),
        },
    )


def _seal_package(
    arguments: argparse.Namespace,
    *,
    workspace: Path,
    gate: WorkspaceMutationGate,
    registry: SecIssuerRegistrySnapshot,
    listing: AdmittedListingTickerAuthority,
    bundle: RecordedEvidenceDocumentBundle,
    entities: tuple[str, ...],
    semantic_relative: str,
    capability_hash: str,
    profile: EvidenceReviewModelProfile | None,
    authority_id: str,
    minimum_entity_coverage: float,
    selection: MatterSelectionPolicy | None,
    facts: dict[str, object],
) -> dict[str, object]:
    """Publish one immutable package and, when asked, bind it into the workspace."""
    package_id = str(
        canonical_hash(
            {
                "registry": registry.registry_hash,
                "listing": listing.authority_hash,
                "documents": bundle.bundle_hash,
                "semantic_path": semantic_relative,
                "semantic_capability": capability_hash,
                "model_profile": None if profile is None else profile.profile_hash,
                "authority_id": authority_id,
                "minimum_entity_coverage": minimum_entity_coverage,
                # Absent for the retained recipe, so its package ids are unchanged.
                **(
                    {"retrieval_recipe": arguments.recipe}
                    if arguments.recipe and arguments.recipe != RECIPE_MINILM_CPU
                    else {}
                ),
                # Absent for the production plan, so every installed id is unchanged.
                **({"matter_selection": selection.selection_id} if selection is not None else {}),
            }
        )
    )
    destination = (workspace / "authority" / "evidence-cro" / package_id).resolve()
    if not destination.is_relative_to(workspace):
        raise ValueError("evidence_review.package_path_escapes_workspace")
    registry_binding = _publish(
        destination / "issuer-registry.json", registry, registry.registry_hash, workspace=workspace
    )
    listing_binding = _publish(
        destination / "listing-authority.json", listing, listing.authority_hash, workspace=workspace
    )
    documents_binding = _publish(
        destination / "recorded-documents.json", bundle, bundle.bundle_hash, workspace=workspace
    )
    manifest = EvidenceReviewWorkspaceManifest.create(
        authority_id=authority_id,
        issuer_registry=registry_binding,
        listing_authority=listing_binding,
        recorded_documents=documents_binding,
        semantic_model_relative_path=semantic_relative,
        semantic_capability_hash=capability_hash,
        model_profile=profile,
        minimum_entity_coverage=minimum_entity_coverage,
        retrieval_recipe=arguments.recipe or RECIPE_MINILM_CPU,
        matter_selection=selection,
    )
    manifest_path = destination / "evidence-review-authority.json"
    published = _publish(manifest_path, manifest, manifest.authority_hash, workspace=workspace)
    binding = ResearchWorkspaceEvidenceReview(
        relative_path=published.relative_path,
        file_sha256=published.file_sha256,
    )
    replaced = None
    if arguments.install:
        current, _installed = update_research_workspace_manifest(
            workspace, lambda current: current.with_bindings(evidence_review=binding), gate=gate
        )
        if current.evidence_review is not None and current.evidence_review != binding:
            replaced = current.evidence_review.relative_path
    return {
        **({"replaced": replaced, "replaced_package_rule": PACKAGE_RULE} if replaced else {}),
        "authority_hash": manifest.authority_hash,
        "package_id": package_id,
        "binding": binding.model_dump(mode="json"),
        **facts,
        "source_registry_hash": registry.registry_hash,
        "admitted_document_count": len(bundle.documents),
        "entity_ids": entities,
        "listing_entries": len(listing.entries),
        "minimum_entity_coverage": minimum_entity_coverage,
        "semantic_capability_hash": capability_hash,
        "model_profile_hash": None if profile is None else profile.profile_hash,
        "installed": bool(arguments.install),
        "matter_selection": None if selection is None else selection.model_dump(mode="json"),
        "claim": "Local evidence package; no model inference or current strategy authority.",
    }


def _matter_selection(arguments: argparse.Namespace) -> MatterSelectionPolicy:
    """The matter selection a fresh package asks every request for, bound
    into the authority hash, the workspace's file identity and each request:
    the integrated selection, the one current method (the production plan
    and the candidate needs allocation were retired by the first-release
    integration). A rebind keeps the installed selection unless one is
    named, and refuses to keep a retired one.
    """
    del arguments
    return MatterSelectionPolicy(
        method=MATTER_SELECTION_INTEGRATED, families=INTEGRATED_FAMILY_SPELLING
    )


def _acquisition_cutoff(arguments: argparse.Namespace, now: datetime) -> datetime | None:
    """The declared, timezone-aware disclosure cutoff, at or before now: the
    filings selected are those the official source accepted by then, whenever
    they are downloaded. Absent, the run's own clock is the cutoff.
    """
    raw = getattr(arguments, "evidence_as_of", None)
    if raw is None:
        return None
    try:
        value = datetime.fromisoformat(str(raw))
    except ValueError as error:
        raise OptionRefused("evidence_review.evidence_as_of_invalid", "evidence_as_of") from error
    if value.tzinfo is None or value.utcoffset() is None:
        raise OptionRefused("evidence_review.evidence_as_of_invalid", "evidence_as_of")
    if value > now:
        raise OptionRefused("evidence_review.evidence_as_of_in_the_future", "evidence_as_of")
    return value.astimezone(UTC)


def _accession_scopes(
    arguments: argparse.Namespace, entities: tuple[str, ...]
) -> dict[str, frozenset[str]]:
    """Named originals for one declared issuer: the selection is exactly those
    accessions as the issuer's own official index holds them.
    """
    raw = tuple(getattr(arguments, "accessions", None) or ())
    if not raw:
        return {}
    if len(entities) != 1:
        raise OptionRefused("evidence_review.accession_scope_requires_one_entity", "accessions")
    accessions = tuple(value.strip() for value in raw)
    if len(set(accessions)) != len(accessions) or len(accessions) > 4:
        raise OptionRefused("evidence_review.accession_scope_invalid", "accessions")
    if any(not re.fullmatch(r"\d{10}-\d{2}-\d{6}", value) for value in accessions):
        raise OptionRefused("evidence_review.accession_scope_invalid", "accessions")
    return {entities[0]: frozenset(accessions)}


def _acquire_sec_sources(
    arguments: argparse.Namespace,
    entities: tuple[str, ...],
    *,
    official_source: OfficialSourceAdmission | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    cutoff: datetime | None = None,
    accession_scopes: dict[str, frozenset[str]] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[Path, str, dict[str, object]]:
    workspace = arguments.workspace.resolve()
    owned = official_source is None
    official_source = official_source or admit_official_source(
        network_consent=bool(arguments.network_consent),
        workspace_root=workspace,
        maximum_document_bytes=arguments.maximum_document_bytes,
    )
    source = official_source.source
    runtime = None
    try:
        if not official_source.network_consent:
            raise ValueError("evidence_review.explicit_sec_network_consent_required")
        if official_source.transport_origin != "INJECTED" and not network_access(workspace).allowed:
            raise ValueError("evidence_review.workspace_network_not_allowed")
        if official_source.refusal_code or source is None:
            raise ValueError(
                official_source.refusal_code or "evidence_review.sec_contact_not_configured"
            )
        if (
            arguments.maximum_documents_per_issuer > official_source.maximum_documents_per_issuer
            or (
                official_source.maximum_document_bytes is not None
                and arguments.maximum_document_bytes > official_source.maximum_document_bytes
            )
        ):
            raise ValueError("evidence_review.budget_exceeds_consent")
        now = clock()
        request = seal_contract(
            AlternativeEvidenceRequest,
            "request_hash",
            ordered_entity_ids=entities,
            evidence_as_of=now if cutoff is None else cutoff,
            acquisition_deadline=now + timedelta(minutes=15),
            evidence_classes=(AlternativeEvidenceClass.SEC_FILING,),
            source_policy=AlternativeEvidenceSourcePolicy(
                maximum_documents_per_issuer=arguments.maximum_documents_per_issuer,
                maximum_document_bytes=arguments.maximum_document_bytes,
            ),
            ttl_seconds=86400,
            mode=AlternativeEvidenceMode.LIVE_OFFICIAL,
        )
        admission = seal_live_setup_admission(request.request_hash, now)
        recipe = arguments.recipe or RECIPE_MINILM_CPU
        runtime = AlternativeEvidenceDocumentIntelligenceRuntime(
            artifact_root=workspace / "runtime" / "artifacts",
            workspace_root=workspace / "runtime" / "evidence-knowledge",
            model_root=_semantic_root(arguments, workspace=workspace, recipe=recipe),
            retrieval_recipe=recipe,
        )
        transport = getattr(source, "_transport", None)
        counters = {
            "http_attempts": "request_count",
            "retries": "retry_count",
            "rate_limited": "rate_limited_count",
            "response_bytes": "response_bytes",
        }
        before = {key: getattr(transport, field, None) for key, field in counters.items()}
        network_before = source.network_call_count
        _registry, snapshot, source_set = runtime.acquire_live(
            request=request,
            admission=admission,
            source=source,
            published_at=now,
            accession_scopes=accession_scopes or None,
            should_cancel=cancelled,
            clock=clock,
        )
        return (
            runtime.artifacts.root,
            source_set.source_set_hash,
            {
                "mode": "LIVE_OFFICIAL",
                "evidence_as_of": request.evidence_as_of.isoformat(),
                "acquired_at": now.isoformat(),
                "network_calls": source.network_call_count - network_before,
                **{
                    key: None if before[key] is None else getattr(transport, field) - before[key]
                    for key, field in counters.items()
                },
                "snapshot_status": snapshot.status.value,
                "snapshot_hash": snapshot.snapshot_hash,
                "source_document_count": len(source_set.documents),
                "limitations": list(snapshot.limitations),
            },
        )
    finally:
        if runtime is not None:
            runtime.close()
        if owned and source is not None:
            source.close()


def _research_universe_listing_ids(workspace: Path, binding_hash: str) -> dict[str, str]:
    # The caller admitted and verified this input through ResearchInputRevisions.
    bundle = read_factor_bundle(workspace, binding_hash, verify=False)
    source = workspace / "research-inputs" / binding_hash / "source"
    resolver = ArtifactResolver(source / "artifacts")
    panel = resolver.load_feature_panel_manifest(
        resolver.feature_panel_manifest_uri(bundle.panel_snapshot_hash)
    )
    revision = str(panel["safe_summary"]["lineage"]["manifest_revision"])
    universe = MarketDataRepository(source).load_universe_manifest_revision(revision)
    return {entry.symbol: entry.listing_id for entry in universe.listings}


def _universe_listing_ids(workspace: Path) -> dict[str, str]:
    """Ticker to listing id, read from the universe the product itself resolves.

    The review joins the window-end book on `listing_id`, and that id is the
    universe's, not a ticker. Sealing the authority with the ticker in both
    fields therefore mapped nothing: every position landed in the mapping gaps
    and the scope came out empty before a single document was read.

    The mapping is taken from the same risk surface and sector map the
    portfolio composition resolves, rather than restated here, because an
    authority bound to a different universe revision than the book would fail
    the same way and be much harder to see.
    """
    installed = installed_authority(workspace, read_research_workspace_manifest(workspace))
    if installed is None:
        artifact_root = workspace / "runtime" / "artifacts"
        surface = _risk_surface(artifact_root)
        classification = _sector_map(
            artifact_root, manifest_revision=surface.epoch.universe_manifest_revision
        )
    else:
        artifact_root, authority = installed
        surface = RiskReturnArtifactStore(artifact_root).load_manifest(
            authority.risk_return_surface_hash
        )
        classification = PanelClosureArtifactStore(ArtifactResolver(artifact_root)).load_model(
            category="sector-maps", content_hash=authority.sector_map_hash, model=SectorRevisionMap
        )
        if (
            surface.surface_hash != authority.risk_return_surface_hash
            or classification.manifest_revision != surface.epoch.universe_manifest_revision
        ):
            raise ValueError("portfolio_application.saved_source_binding_mismatch")
    return {entry.provider_symbol: entry.listing_id for entry in classification.entries}


DEFAULT_AUTHORITY_ID = "local-web-evidence-authority"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--rebind-installed",
        action="store_true",
        help=(
            "Re-seal the installed package's listing authority over the workspace universe; "
            "no source, no acquisition. Authority id, floor, model profile and matter "
            "selection stay unless given."
        ),
    )
    parser.add_argument("--source-artifact-root", type=Path)
    parser.add_argument(
        "--source-knowledge-root",
        type=Path,
        help=(
            "the source's evidence Workspace, holding the bytes a reference-form source "
            "set names; defaults to the runtime/evidence-knowledge beside the artifact root"
        ),
    )
    parser.add_argument("--source-set-hash")
    parser.add_argument(
        "--acquire-sec",
        action="store_true",
        help="Acquire a declared issuer scope from existing official SEC owners; no data API key.",
    )
    parser.add_argument(
        "--entities", nargs="+", help="Explicit issuer/ticker scope, at most eight."
    )
    parser.add_argument("--maximum-documents-per-issuer", type=int, default=3)
    parser.add_argument(
        "--maximum-document-bytes",
        type=int,
        default=DEFAULT_SOURCE_DOCUMENT_BYTES,
        help="Per-document response cap, within the source policy's admitted range; "
        "the source policy's own default unless declared; declared before any "
        "request, never raised after a failure.",
    )
    parser.add_argument(
        "--evidence-as-of",
        help="Timezone-aware disclosure cutoff (ISO 8601) at or before now; the filings "
        "selected are those the official source accepted by then. Defaults to the "
        "run's clock.",
    )
    parser.add_argument(
        "--accessions",
        nargs="+",
        help="Exact accession numbers to acquire for the one declared issuer, as its own "
        "official submissions index holds them; other filings are deferred.",
    )
    parser.add_argument("--network-consent", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check without installing: an SEC acquisition's setup, input and model "
        "prerequisites, acquiring nothing, or a recorded import's records, issuers, documents "
        "and floor, offline.",
    )
    parser.add_argument(
        "--semantic-model",
        type=Path,
        help="The retained recipe's pack directory inside the workspace; without it the "
        "recipe is bound from the application model store (V208).",
    )
    parser.add_argument(
        "--recipe",
        choices=SUPPORTED_RECIPES,
        help=f"Retrieval recipe; defaults to {RECIPE_MINILM_CPU!r}. Other recipes are bound "
        "from the application model store (install_retrieval_pack.py).",
    )
    parser.add_argument(
        "--model-store", type=Path, help="Application model store override for --recipe."
    )
    parser.add_argument(
        "--research-input-id", help="Admitted research input family for an authored book."
    )
    parser.add_argument(
        "--research-input-hash", help="Exact sealed input binding; never inferred from latest data."
    )
    parser.add_argument(
        "--authority-id", help=f"Defaults to {DEFAULT_AUTHORITY_ID!r} (installed id on rebind)."
    )
    parser.add_argument(
        "--model-name", help="Optional managed analyst/reviewer model; omit for native analysis."
    )
    parser.add_argument("--deepseek-base-url", help="Endpoint for the optional managed model.")
    parser.add_argument("--analyst-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--review-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--minimum-entity-coverage",
        type=float,
        help="Defaults to 0.60 (the installed floor on rebind).",
    )
    parser.add_argument(
        "--matter-selection",
        choices=(MATTER_SELECTION_INTEGRATED,),
        help=(
            "The matter selection the package asks every request for, bound into the "
            "authority hash: the integrated selection, the one current method and a fresh "
            "install's default (the path the coverage ledger and the time view read). A "
            "rebind keeps the installed selection unless one is named; an authority "
            "installed under the retired production plan or candidate selection is "
            "rebound only by naming it."
        ),
    )
    parser.add_argument("--install", action="store_true")
    return parser


def _setup_entry() -> list[str]:
    """This setup as the person runs it: the checkout's script, or the installed module."""
    if (PLAYPEN_ROOT / "pyproject.toml").is_file():
        return [sys.executable, "scripts/materialize_evidence_cro_authority.py"]
    return [
        sys.executable,
        "-m",
        "alphalattice.control.product_host.composition.evidence_authority_setup",
    ]


def _acquisition_command(
    arguments: argparse.Namespace,
    entities: Sequence[str],
    evidence_as_of: str | None,
    *,
    preflight: bool,
    host: bool = False,
) -> str:
    """The official acquisition of these issuers at this cutoff, as the person types it.

    Bound to the workspace, the research input and the scope given: its read-only check, or
    the acquisition itself with what shapes the package and its installation as asked (V587).
    The person's consent is carried only where they gave it.
    """
    parts = ["--acquire-sec"]
    if preflight:
        parts.append("--preflight")
    shaping: tuple[tuple[str, object], ...] = (
        ("--authority-id", arguments.authority_id),
        ("--minimum-entity-coverage", arguments.minimum_entity_coverage),
        ("--matter-selection", arguments.matter_selection),
        ("--model-name", arguments.model_name),
        ("--deepseek-base-url", arguments.deepseek_base_url),
        *(
            (
                ("--analyst-timeout-seconds", arguments.analyst_timeout_seconds),
                ("--review-timeout-seconds", arguments.review_timeout_seconds),
            )
            if arguments.model_name
            else ()
        ),
    )
    for flag, value in (
        ("--research-input-id", arguments.research_input_id),
        ("--research-input-hash", arguments.research_input_hash),
        ("--recipe", arguments.recipe),
        ("--model-store", arguments.model_store),
        ("--semantic-model", arguments.semantic_model),
        ("--evidence-as-of", evidence_as_of),
        ("--maximum-documents-per-issuer", arguments.maximum_documents_per_issuer),
        ("--maximum-document-bytes", arguments.maximum_document_bytes),
        *(() if preflight else shaping),
    ):
        if value is not None:
            parts += [flag, str(value)]
    if entities:
        parts += ["--entities", *entities]
    if arguments.accessions:
        parts += ["--accessions", *arguments.accessions]
    if arguments.network_consent:
        parts.append("--network-consent")
    if not preflight and arguments.install:
        parts.append("--install")
    return _typed(arguments, parts, host)


def _import_command(arguments: argparse.Namespace, root: Path, host: bool = False) -> str:
    """The recorded import's own check at this root, as the person types it (V590).

    Bound to the workspace, the research input, the source set and the issuers given: it reads
    and judges what the install would read, offline, and installs nothing.
    """
    parts: list[str] = []
    for flag, value in (
        ("--research-input-id", arguments.research_input_id),
        ("--research-input-hash", arguments.research_input_hash),
        ("--source-artifact-root", root),
        ("--source-set-hash", arguments.source_set_hash),
        ("--source-knowledge-root", arguments.source_knowledge_root),
        ("--recipe", arguments.recipe),
        ("--model-store", arguments.model_store),
        ("--semantic-model", arguments.semantic_model),
        ("--minimum-entity-coverage", arguments.minimum_entity_coverage),
    ):
        if value is not None:
            parts += [flag, str(value)]
    if arguments.entities:
        parts += ["--entities", *arguments.entities]
    parts.append("--preflight")
    return _typed(arguments, parts, host)


def _typed(arguments: argparse.Namespace, options: list[str], host: bool) -> str:
    """The setup with these options as the person types it: through the running Host, which
    holds the workspace, or the script while no Host runs."""
    if host:
        setup = "--setup=" + shlex.join(options)
        workspace = str(arguments.workspace)
        return join(
            [*command_prefix(), "--workspace", workspace, "evidence", "install", setup], shell()
        )
    return join([*_setup_entry(), "--workspace", str(arguments.workspace), *options], shell())


_RUNTIME_REFUSALS = (
    "evidence_review.semantic_capability_not_ready",
    "evidence_review.retrieval_environment_not_loaded",
)
"""The retrieval runtime or its packs are not ready: filling them is the way on."""


def setup_refusal(
    arguments: argparse.Namespace, error: Exception, *, script: bool = False
) -> dict[str, Any]:
    """A setup refusal whole.

    Its code and words, its context, the option it refused on and the commands of its way on:
    the running Host's route, or the script's where the script runs with no Host (`script`).
    """
    payload = setup_failure(error)
    payload["context"] = {
        "workspace": str(arguments.workspace),
        "source": "SEC"
        if arguments.acquire_sec
        else ("REBIND_INSTALLED" if arguments.rebind_installed else "RECORDED_IMPORT"),
        "issuers": arguments.entities or [],
        "research_input_id": arguments.research_input_id,
        "research_input_hash": arguments.research_input_hash,
        "source_set_hash": arguments.source_set_hash,
        "network_consent": arguments.network_consent,
    }
    # A script that met a running Host is sent to that Host's route.
    host = not script or str(error) == "workspace_runtime.writer_already_owned"
    if arguments.acquire_sec:
        payload["next_requests"] = {"network": {"operation": "NETWORK_ACCESS"}}
    payload["next_commands"] = _ways_on(arguments, error, host)
    if isinstance(error, OptionRefused):
        # The option it refused on and what that option must hold, as the offer states it
        # (V591, V592); a file under a root option, by its place there (V590).
        payload["location"] = {
            "option": error.option,
            **({"file": error.file} if isinstance(error, SourceFileUnavailable) else {}),
            "expected": SETUP_OPTIONS[error.option],
        }
    if (
        isinstance(error, SourceCoverageRefused)
        and arguments.acquire_sec
        and error.failed
        and len(error.entities) <= 8
    ):
        # The way on: the same official acquisition of the package's issuers at the one
        # cutoff they were counted at -- what failed is fetched again and a quiet issuer
        # stays uncounted -- checked first (V587).
        cutoff, entities = error.evidence_as_of.isoformat(), error.entities
        payload["next_commands"] = {
            "preflight": _acquisition_command(
                arguments, entities, cutoff, preflight=True, host=host
            ),
            "acquire": _acquisition_command(
                arguments, entities, cutoff, preflight=False, host=host
            ),
        }
    if (
        arguments.acquire_sec
        and payload["failure_code"] == "evidence_review.workspace_network_not_allowed"
    ):
        permission = network_access(arguments.workspace).body(for_refusal=True)
        payload.update(refusal_words(payload["failure_code"], workspace=arguments.workspace))
        payload["network_access"] = permission
        payload["next_requests"] = {
            "network": {"operation": "NETWORK_ACCESS"},
            **cast(dict[str, object], permission["next_requests"]),
        }
    return payload


def _ways_on(arguments: argparse.Namespace, error: Exception, host: bool) -> dict[str, str]:
    """A refusal's commands: what fills a retrieval runtime that is not ready, else the mode's
    check (V590). An acquisition's read-only preflight keeps its exact binding, scope and
    explicit consent; a recorded import's way on is its own check, offline."""
    if str(error) in _RUNTIME_REFUSALS:
        # Never the same check again: the runtime's fill and the packs' status come first.
        return {
            "environment": join(fill_command(), shell()),
            "packs": join(pack_command("--status"), shell()),
        }
    if arguments.acquire_sec:
        cutoff, entities = arguments.evidence_as_of, arguments.entities or ()
        return {
            "preflight": _acquisition_command(
                arguments, entities, cutoff, preflight=True, host=host
            )
        }
    if arguments.source_artifact_root is not None and arguments.source_set_hash:
        root = arguments.source_artifact_root
        if isinstance(error, SourceFileUnavailable) and error.option == "source_artifact_root":
            root = _holding_root(root.resolve(), arguments.source_set_hash) or root
        return {"preflight": _import_command(arguments, root, host)}
    entry = [*command_prefix(), "evidence", "install"] if host else _setup_entry()
    return {"help": join([*entry, "--help"], shell())}


def main(argv: list[str] | None = None) -> int:
    """Run the declared command and return its exit status."""
    arguments = _parser().parse_args(argv)
    try:
        result = materialize(arguments)
    except Exception as error:
        print(json.dumps(setup_refusal(arguments, error, script=True)))
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
