"""Compose immutable Alpha evidence and local Market inputs for EW research."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any, get_args

import annotated_types
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import yaml  # type: ignore[import-untyped]

from alphalattice.capabilities.portfolio_backtesting.clocks import EveryFormationClock
from alphalattice.capabilities.portfolio_backtesting.contracts import RebalanceClock
from alphalattice.capabilities.portfolio_inputs.session_marks import (
    resolve_session_mark_availability,
)
from alphalattice.capabilities.portfolio_inputs.tradability.readback import (
    read_tradability_matrices,
)
from alphalattice.capabilities.portfolio_inputs.tradability.surface import (
    HistoricalTradabilityBuilder,
    PortfolioTradabilityMarketInputs,
    TradabilityBuildCancelled,
)
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.execution import _sector_by_listing_id
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    confined,
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
)
from alphalattice.foundation.feature_engine.panels.reader import FeaturePanelReader
from alphalattice.foundation.market_data_ops.publication.session_marks import (
    SessionMarkArtifactStore,
    publish_market_session_marks,
)
from alphalattice.foundation.market_data_ops.storage.duckdb import MarketDataRepository
from alphalattice.investment.alpha_research.experiments.development_artifacts import (
    AlphaDevelopmentArtifactStore,
)
from alphalattice.investment.alpha_research.experiments.development_contracts import (
    AlphaDevelopmentExecutionReceipt,
    canonical_score_value_identity,
)
from alphalattice.investment.alpha_research.experiments.development_evidence import (
    AlphaDevelopmentReceiptReader,
    alpha_development_receipt_handle,
)
from alphalattice.investment.alpha_research.experiments.score_rows import (
    DevelopmentScoreMatrix,
    read_experiment_score_matrix,
)
from alphalattice.investment.portfolio_strategy_lab.application.research_experiment import (
    KIND,
    TRANCHE_FIELDS,
    PortfolioExperimentCancelled,
    PortfolioExperimentExecutor,
    PortfolioExperimentInputs,
    PortfolioExperimentSource,
    PortfolioExperimentSpec,
    ResearchUniversePolicy,
    resolve_research_universe,
)
from alphalattice.investment.portfolio_strategy_lab.campaign.authority import (
    load_stage_six_covariance,
    resolve_portfolio_state_transition_binding,
    resolve_reference_mark_lane,
)
from alphalattice.investment.portfolio_strategy_lab.contracts import (
    PortfolioPolicyFamily,
    PortfolioPolicySpec,
)
from alphalattice.investment.portfolio_strategy_lab.inputs.shared_lanes import (
    outcome_returns,
    sector_exposure_lanes,
)
from alphalattice.investment.portfolio_strategy_lab.policies.lifecycle_research import (
    LocalQAOutcomeBinding,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
)


@dataclass(frozen=True)
class PreparedPortfolioHandoff:
    """Retain exact Alpha scores, normalized Portfolio declaration and preview.

    Retain exact upstream Alpha scores, normalized Portfolio declaration and execution preview.
    """

    workspace: Path
    source_workspace: Path
    source: PortfolioExperimentSource
    document: dict[str, Any]
    executor: PortfolioExperimentExecutor
    preview: dict[str, Any]

    def workflow(self) -> ResearchProgramWorkflow:
        """Compose the exact Portfolio workflow with its retained executor/compiler and source root.

        Returns:
            Research workflow bound to the declared source and caller-owned workspace.
        """
        return build_research_program_workflow(
            workspace=self.source_workspace,
            workspace_root=self.workspace,
            executors=(self.executor,),
            compilers=(self.executor.compiler,),
            verifier_kinds=(KIND,),
        )


@dataclass(frozen=True)
class PortfolioRiskLink:
    """A completed Risk study a Portfolio study weighs by, as its verified readback shows it."""

    task_id: str
    program_hash: str
    surface_hash: str
    evidence_root: Path
    input_binding_hash: str
    panel_snapshot_hash: str
    universe_revision: str
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]


@dataclass(frozen=True)
class PortfolioMarketContext:
    """A bound research market question, without an invented Alpha candidate or parent."""

    input_binding_hash: str
    universe_revision: str
    panel_snapshot_hash: str
    outcome_snapshot_hash: str
    formation_sessions: tuple[date, ...]
    ordered_listing_ids: tuple[str, ...]
    clock: RebalanceClock = field(default_factory=EveryFormationClock)


@dataclass(frozen=True)
class LocalQAOutcomeRows:
    """Retain exact locally observed QA outcome rows and their explicit binding."""

    binding: LocalQAOutcomeBinding
    rows: pa.Table


_SOLVER_BACKED = frozenset(
    {
        PortfolioPolicyFamily.TOP_K_MINIMUM_VARIANCE,
        PortfolioPolicyFamily.TOP_K_SCORE_RISK_COST,
        PortfolioPolicyFamily.SECTOR_DEVIATION_PENALTY,
    }
)
"""The catalog policies whose weights a solver decides."""


def hold_under_scored_formations(
    scores: DevelopmentScoreMatrix, selected: int
) -> tuple[DevelopmentScoreMatrix, tuple[date, ...]]:
    """The scores a book rebalances on, and the formations it holds for too few scored names.

    A rebalance selects ``selected`` names from those the candidate scored. On a formation the
    candidate scored fewer -- a feature unavailable across the universe, say -- the book holds
    its drifted positions, as on an embargo session, rather than select from a pool too small
    for it. A book starts and ends on a scored formation, so either end under-scored is
    refused by its session.

    Args:
        scores: The candidate's published scores on its formation sessions.
        selected: How many names one rebalance selects.

    Returns:
        The scores on the formations the book rebalances, and those it holds.

    Raises:
        AuthoringError: The book's first or last formation is under-scored.
    """
    if selected > len(scores.listings):
        # A book selecting more names than its universe holds is refused by that bound when its
        # program compiles (`top_k_exceeds_universe` with its maximum, or the tranche book's
        # `exit_rank_exceeds_universe`), never held.
        return scores, ()
    counts = np.count_nonzero(scores.available, axis=1)
    short = tuple(
        session for session, count in zip(scores.sessions, counts, strict=True) if count < selected
    )
    if not short:
        return scores, ()
    for end in (scores.sessions[0], scores.sessions[-1]):
        if end in short:
            raise AuthoringError(f"portfolio_research.book_end_under_scored:{end}")
    kept = np.flatnonzero(counts >= selected)
    matrix = scores.scores[kept]
    available = scores.available[kept]
    matrix.setflags(write=False)
    available.setflags(write=False)
    held = DevelopmentScoreMatrix(
        tuple(scores.sessions[i] for i in kept),
        scores.listings,
        matrix,
        available,
        scores.refs,
        canonical_score_value_identity(matrix),
    )
    return held, short


def prepare_portfolio_handoff(
    *,
    workspace: Path,
    input_binding_hash: str,
    alpha_task_id: str,
    alpha_program_hash: str,
    alpha_document: Mapping[str, Any],
    alpha_receipt_handle: str,
    candidate_id: str,
    document: dict[str, Any] | None = None,
    cancellation: Callable[[], bool] = lambda: False,
    alpha_receipt: AlphaDevelopmentExecutionReceipt | None = None,
    risk: PortfolioRiskLink | None = None,
) -> PreparedPortfolioHandoff:
    """Prepare the Portfolio declaration over one verified Alpha receipt.

    ``alpha_receipt`` is the receipt the caller's own verified readback of
    ``alpha_receipt_handle`` walked and proved in this same request; it is
    taken as that walk's value, and its identity must be the handle's. Left
    unset, the handoff walks the graph itself.
    """
    bundle = read_factor_bundle(workspace, input_binding_hash)
    source_workspace, artifact_root = factor_input_paths(workspace, input_binding_hash)
    alpha_root = (
        confined(workspace, str(alpha_document["experiment"]["output_workspace"]))
        / "alpha-development"
    )
    if alpha_receipt is None:
        receipt = AlphaDevelopmentReceiptReader(alpha_root).load(alpha_receipt_handle)
    elif alpha_receipt.receipt_hash != alpha_development_receipt_handle(alpha_receipt_handle):
        raise AuthoringError("portfolio_research.alpha_receipt_handle_mismatch")
    else:
        receipt = alpha_receipt
    if receipt.program_hash != alpha_program_hash:
        raise AuthoringError("portfolio_research.alpha_program_mismatch")
    if receipt.target_recipe_binding.causal_outcome_snapshot_hash != bundle.outcome_snapshot_hash:
        raise AuthoringError("portfolio_research.one_session_outcome_required")
    scores = read_experiment_score_matrix(
        AlphaDevelopmentArtifactStore(alpha_root), receipt, candidate_id
    )
    outcome_reader = CausalExecutionOutcomeDevelopmentReader(artifact_root)
    outcome = outcome_reader.load_manifest(bundle.outcome_snapshot_hash)
    seal = outcome_reader.resolve_method_seal(outcome.snapshot_hash)
    if seal.disposition != "METHOD_BOUND":
        raise AuthoringError("portfolio_research.outcome_method_unbound")
    outcome_ref = outcome_reader.manifest_uri(outcome.snapshot_hash)
    schedule = tuple(outcome_reader.read_development_schedule(outcome_ref).to_pylist())
    selected_schedule = tuple(
        v for v in schedule if scores.sessions[0] <= v["formation_session"] <= scores.sessions[-1]
    )
    sessions = tuple(v["formation_session"] for v in selected_schedule)
    parent = ResearchExperimentEnvelope.create(**alpha_document["experiment"])
    if (
        not sessions
        or sessions[0] != scores.sessions[0]
        or sessions[-1] != scores.sessions[-1]
        or not set(scores.sessions) <= set(sessions)
        or max(v["holding_end_session"] for v in selected_schedule) > parent.sessions.as_of.session
    ):
        raise AuthoringError("portfolio_research.economic_calendar_unavailable")
    declaration = deepcopy(dict(document or {}))
    if declaration and set(declaration) != {"experiment", "portfolio"}:
        raise AuthoringError("portfolio_research.document_sections_invalid")
    spec = PortfolioExperimentSpec.model_validate(
        declaration.get(
            "portfolio",
            {
                "alpha_task_id": alpha_task_id,
                "candidate_id": candidate_id,
                "unavailable_return_policy": "quarantine_listings",
            },
        )
    )
    if spec.alpha_task_id != alpha_task_id or spec.candidate_id != candidate_id:
        raise AuthoringError("portfolio_research.alpha_selection_mismatch")
    # A catalog policy runs in place of the tranche book, whose own fields then stay unset.
    if spec.policy is not None and any(
        getattr(spec, name) != PortfolioExperimentSpec.model_fields[name].default
        for name in TRANCHE_FIELDS
    ):
        raise AuthoringError("portfolio_research.policy_book_conflict")
    # An inverse-volatility rule weighs by a Risk study's lane and a catalog policy by its
    # covariance; equal weight reads none, so a linked study it would not read is refused
    # rather than bound unread.
    reads_risk = spec.weight_rule != "ew" or spec.policy is not None
    if reads_risk and spec.risk_task_id is None:
        raise AuthoringError("portfolio_research.risk_study_required")
    if not reads_risk and spec.risk_task_id is not None:
        raise AuthoringError("portfolio_research.risk_study_unread")
    if (risk is None) != (spec.risk_task_id is None) or (
        risk is not None and risk.task_id != spec.risk_task_id
    ):
        raise AuthoringError("portfolio_research.risk_selection_mismatch")
    experiment = deepcopy(dict(alpha_document["experiment"]))
    experiment.pop("envelope_hash", None)
    experiment.update(
        kind=KIND,
        # Portfolio reads saved Alpha scores and this bound market/outcome input,
        # not the upstream model's optional composed Feature surface. Its exact
        # Alpha receipt and score hashes below retain that Feature provenance.
        data_snapshot_handle=bundle.panel_snapshot_hash,
        sessions={
            "start": str(sessions[0]),
            "end": str(sessions[-1]),
            "as_of": parent.sessions.as_of.model_dump(mode="json"),
        },
    )
    authored = declaration.get("experiment", {})
    for key in ("budget", "determinism"):
        if key in authored:
            experiment[key] = authored[key]
    experiment["output_workspace"] = "managed"
    experiment["baseline_workspace"] = f"research-inputs/{input_binding_hash}/source"
    key = canonical_hash(
        {
            "experiment": experiment,
            "portfolio": spec.model_dump(mode="json"),
            "input": input_binding_hash,
        }
    )
    experiment["output_workspace"] = f"research-experiments/portfolio-{key}"
    if authored and (
        set(authored) - set(experiment) - {"envelope_hash"}
        or any(
            authored.get(k) not in ("managed", v)
            if k in {"output_workspace", "baseline_workspace"}
            else authored.get(k) != v
            for k, v in experiment.items()
            if k != "envelope_hash"
        )
    ):
        raise AuthoringError("portfolio_research.authority_field_mismatch")
    declaration = {"experiment": experiment, "portfolio": spec.model_dump(mode="json")}
    envelope = ResearchExperimentEnvelope.create(**experiment)
    authority = WorkspaceResearchAuthorityResolver(workspace=source_workspace).resolve(envelope)
    if authority.ordered_listing_ids != scores.listings or authority.sessions != sessions:
        raise AuthoringError("portfolio_research.alpha_market_axis_mismatch")
    if risk is not None:
        # The Risk study reads the same research input, Panel and universe as the Alpha scores,
        # and covers every formation and listing the book weighs.
        if (
            risk.input_binding_hash != input_binding_hash
            or risk.panel_snapshot_hash != bundle.panel_snapshot_hash
            or risk.universe_revision != authority.universe_revision_sha256
        ):
            raise AuthoringError("portfolio_research.risk_input_mismatch")
        if not set(sessions) <= set(risk.formation_sessions):
            raise AuthoringError("portfolio_research.risk_session_axis_incomplete")
        if not set(scores.listings) <= set(risk.ordered_listing_ids):
            raise AuthoringError("portfolio_research.risk_listing_axis_incomplete")
    # A rebalance selects the book's names from those the candidate scored; a formation it
    # under-scored is held, as an embargo session is.
    scores, under_scored = hold_under_scored_formations(
        scores, spec.top_k if spec.policy is None else spec.policy.top_k
    )
    source = PortfolioExperimentSource.create(
        alpha_task_id=alpha_task_id,
        alpha_program_hash=alpha_program_hash,
        alpha_receipt_hash=receipt.receipt_hash,
        candidate_id=candidate_id,
        target_recipe_id=receipt.target_recipe_binding.target_recipe_id,
        target_binding_hash=receipt.target_recipe_binding.binding_hash,
        foundation_admission_hash=alpha_document["alpha"].get("foundation_admission_hash"),
        input_binding_hash=input_binding_hash,
        panel_snapshot_hash=bundle.panel_snapshot_hash,
        outcome_snapshot_hash=bundle.outcome_snapshot_hash,
        universe_revision=authority.universe_revision_sha256,
        ordered_listing_ids=scores.listings,
        formation_sessions=sessions,
        score_sessions=scores.sessions,
        score_refs=scores.refs,
        score_value_hash=scores.value_hash,
        **(
            {}
            if risk is None
            else {
                "risk_task_id": risk.task_id,
                "risk_program_hash": risk.program_hash,
                "risk_surface_hash": risk.surface_hash,
            }
        ),
    )
    market_key = canonical_hash(
        {
            "input": source.input_binding_hash,
            "outcome": source.outcome_snapshot_hash,
            "sessions": sessions,
            "listings": scores.listings,
        }
    )
    market_root = confined(workspace, f"artifacts/research-market-inputs/{market_key}")
    executor = PortfolioExperimentExecutor(
        source,
        lambda output: prepare_portfolio_market_inputs(
            source=source,
            source_workspace=source_workspace,
            artifact_root=artifact_root,
            output=market_root,
            scores=scores,
            schedule=selected_schedule,
            cancellation=cancellation,
            spec=spec,
        ),
        cancellation=cancellation,
        risk_allocation=None
        if risk is None
        else lambda: load_stage_six_covariance(
            evidence_root=risk.evidence_root, surface_hash=risk.surface_hash
        ).allocation_projections(
            sessions=source.formation_sessions, listing_ids=source.ordered_listing_ids
        ),
        risk_covariance=None
        if risk is None
        else lambda formations: load_stage_six_covariance(
            evidence_root=risk.evidence_root, surface_hash=risk.surface_hash
        ).covariance_lane(sessions=formations, listing_ids=source.ordered_listing_ids),
    )
    preview = {
        "statistical_start": str(sessions[0]),
        "statistical_end": str(sessions[-1]),
        "statistical_session_count": len(sessions),
        "score_session_count": len(scores.sessions),
        "hold_session_count": len(sessions) - len(scores.sessions),
        "listing_count": len(scores.listings),
        "expected_numerical_calls": len(sessions) + 1,
        "fit_calls": 0,
        # At least one solve at each scored formation, for a solver-backed policy.
        "solver_calls": len(scores.sessions)
        if spec.policy is not None and spec.policy.family in _SOLVER_BACKED
        else 0,
        "interval_semantics": "FULL_ALPHA_DEVELOPMENT_SUPPORT_WITH_EMBARGO_HOLDS",
        # The holds among them where the candidate scored fewer names than the book selects.
        **(
            {
                "under_scored_hold_count": len(under_scored),
                "first_under_scored_hold": str(under_scored[0]),
            }
            if under_scored
            else {}
        ),
        "source_hash": source.source_hash,
        "risk_disposition": "NOT_ADMITTED_NOT_USED_FOR_WEIGHTS"
        if risk is None
        else "INVERSE_VOLATILITY_FROM_THE_LINKED_RISK_STUDY"
        if spec.policy is None
        else "COVARIANCE_FROM_THE_LINKED_RISK_STUDY",
    }
    return PreparedPortfolioHandoff(
        workspace, source_workspace, source, declaration, executor, preview
    )


_POLICY_LABELS = {
    "top_k": "Names held",
    "maximum_weight": "Largest weight per name",
    "risk_aversion": "Risk aversion",
    "turnover_regularization": "Turnover charge",
    "sector_deviation_penalty": "Sector deviation charge",
}


def _policy_controls(
    policy: PortfolioPolicySpec | None, listing_count: int
) -> list[dict[str, object]]:
    """A catalog policy's family and each family's fields, read from the catalog's models.

    Each field shows while the document's family is one that declares it (`when`), with the
    model's own bounds and words, so the draft offers what PLAN admits and no rule of its
    own. A name count is bounded by the input's names.

    Args:
        policy: The declared policy, or None for the tranche book.
        listing_count: The names on the draft's input.

    Returns:
        The family control, then one control per field.
    """
    models = get_args(get_args(PortfolioPolicySpec)[0])
    families: dict[str, list[str]] = {}
    fields: dict[str, Any] = {}
    for model in models:
        family = str(model.model_fields["family"].default.value)
        for name, info in model.model_fields.items():
            if name != "family":
                families.setdefault(name, []).append(family)
                fields.setdefault(name, info)
    controls: list[dict[str, object]] = [
        {
            "path": ["portfolio", "policy", "family"],
            "label": "Catalog policy",
            "type": "select",
            "options": [str(model.model_fields["family"].default.value) for model in models],
            "value": None if policy is None else str(policy.family.value),
            "help": (
                "A catalog policy runs in place of the tranche book, on the linked Risk "
                "study's covariance; the tranche book's fields then decide nothing."
            ),
        }
    ]
    for name, info in fields.items():
        bounds: dict[str, object] = {}
        for item in info.metadata:
            if isinstance(item, annotated_types.Ge):
                bounds["min"] = item.ge
            elif isinstance(item, annotated_types.Gt):
                bounds.update(min=item.gt, min_exclusive=True)
            elif isinstance(item, annotated_types.Le):
                bounds["max"] = item.le
        integer = info.annotation is int
        if integer:
            bounds["max"] = listing_count
        controls.append(
            {
                "path": ["portfolio", "policy", name],
                "label": _POLICY_LABELS.get(name, name.replace("_", " ").capitalize()),
                "type": "number",
                "when": [{"path": ["portfolio", "policy", "family"], "values": families[name]}],
                **bounds,
                "step": 1 if integer else "any",
                "value": getattr(policy, name, None),
                "help": info.description,
            }
        )
    return controls


def portfolio_draft(
    prepared: PreparedPortfolioHandoff, risk_studies: tuple[str, ...] = ()
) -> dict[str, object]:
    """The Portfolio draft a published Alpha study hands on, with its controls.

    Args:
        prepared: The prepared handoff.
        risk_studies: The completed Risk studies on the draft's research input, which the
            Risk link offers.

    Returns:
        The draft answer.
    """
    from alphalattice.investment.portfolio_strategy_lab.policies.tranche_book import (
        EXIT_RANK_MULTIPLE_RANGE,
        TOP_K_RANGE,
        TRANCHES_RANGE,
    )

    # Execution paths are derived by PLAN and change with authored parameters.
    # An editable draft must not carry the old derived path as caller authority.
    document = deepcopy(prepared.document)
    document["experiment"].update(output_workspace="managed", baseline_workspace="managed")
    spec = PortfolioExperimentSpec.model_validate(document["portfolio"])
    controls = [
        {
            "path": ["portfolio", name],
            "label": label,
            "type": "number",
            "min": bounds[0],
            "max": bounds[1],
            "step": 1,
            "value": getattr(spec, name),
        }
        for name, label, bounds in (
            ("top_k", "Names per sleeve", TOP_K_RANGE),
            ("tranches", "Sleeve count", TRANCHES_RANGE),
            (
                "exit_rank",
                "Exit rank",
                (
                    spec.top_k,
                    min(
                        len(prepared.source.ordered_listing_ids),
                        int(spec.top_k * EXIT_RANK_MULTIPLE_RANGE[1]),
                    ),
                ),
            ),
        )
    ]
    controls.append(
        {
            "path": ["portfolio", "cost_bps_per_side"],
            "label": "Cost (bps per side)",
            "type": "text",
            "value": spec.cost_bps_per_side,
        }
    )
    # The weight rule and the Risk study it reads.
    controls.append(
        {
            "path": ["portfolio", "weight_rule"],
            "label": "Weight rule",
            "type": "select",
            "options": ["ew", "iv1", "iv2"],
            "value": spec.weight_rule,
            "help": (
                "Equal weight (`ew`), or inverse volatility (`iv1`) or inverse variance "
                "(`iv2`) from the linked Risk study."
            ),
        }
    )
    controls.append(
        {
            "path": ["portfolio", "risk_task_id"],
            "label": "Risk study",
            "type": "select",
            "options": list(risk_studies),
            "value": spec.risk_task_id,
            "help": (
                "A completed Risk study on this research input: an `iv` rule's per-name "
                "volatility, or a catalog policy's covariance. Equal weight with no policy "
                "reads none."
            ),
        }
    )
    controls.extend(_policy_controls(spec.policy, len(prepared.source.ordered_listing_ids)))
    controls.append(
        {
            "path": ["portfolio", "unavailable_return_policy"],
            "label": "Unavailable return handling",
            "type": "select",
            "options": ["quarantine_listings", "require_complete"],
            "value": spec.unavailable_return_policy,
            "help": (
                "Quarantine affected names for this observed research window; "
                "keep source data and disclose the reduced Portfolio universe."
            ),
        }
    )
    return {
        "status": "PORTFOLIO_DRAFT_READY",
        "document": document,
        "yaml": yaml.safe_dump(document, sort_keys=False),
        "controls": controls,
        "input_binding_hash": prepared.source.input_binding_hash,
        "execution_preview": prepared.preview,
        "numerical_call_count": 0,
        # The draft's own declaration: no longer equal weight only, and without a Risk model
        # until it links a Risk study.
        "limitations": [
            "POST_OBSERVED_DEVELOPMENT_NOT_INDEPENDENT_VALIDATION",
            *(["NO_RISK_MODEL"] if spec.risk_task_id is None else []),
            "NO_STRATEGY_ACTIVATION",
        ],
    }


def prepare_portfolio_market_inputs(
    *,
    source: PortfolioExperimentSource | PortfolioMarketContext,
    source_workspace: Path,
    artifact_root: Path,
    output: Path,
    scores: DevelopmentScoreMatrix,
    schedule: tuple[dict[str, Any], ...],
    cancellation: Callable[[], bool],
    spec: ResearchUniversePolicy,
    capacity: Callable[[int], None] = lambda _bytes: None,
    local_qa: LocalQAOutcomeRows | None = None,
) -> PortfolioExperimentInputs:
    # All source bytes were independently verified by the input-bundle owner.
    """Assemble exact Portfolio opportunity, execution, outcome and mark lanes through their owners.

    Decision eligibility uses historical membership and declared quarantine. The passive benchmark
    uses the same population, without filtering members by daily realized outcomes.

    Args:
        source: Exact retained source authority or market context.
        source_workspace: Verified source workspace copy.
        artifact_root: Exact source artifact root.
        output: Caller-owned output artifact root.
        scores: Exact upstream development score matrix.
        schedule: Owner-derived formation/entry/holding-end schedule.
        cancellation: Explicit cancellation callback.
        spec: Explicit research universe policy.
        capacity: Explicit storage reservation callback.
        local_qa: Optional exactly bound local observed outcome rows.

    Returns:
        Immutable Portfolio inputs, lineage identities and explicit data exclusions.

    Raises:
        AuthoringError: Outcome method or exact QA row/axis/value bindings differ.
        PortfolioExperimentCancelled: Tradability preparation cancels at an immutable chunk
            boundary.
    """
    market = MarketDataRepository(source_workspace)
    manifest = market.load_universe_manifest_revision(source.universe_revision)
    sessions, listings = source.formation_sessions, source.ordered_listing_ids
    reader = CausalExecutionOutcomeDevelopmentReader(artifact_root)
    outcome = reader.load_manifest(source.outcome_snapshot_hash)
    seal = reader.resolve_method_seal(outcome.snapshot_hash)
    if seal.disposition != "METHOD_BOUND":
        raise AuthoringError("portfolio_research.outcome_method_unbound")
    if local_qa is not None and (
        local_qa.binding.input_binding_hash != source.input_binding_hash
        or local_qa.binding.formation_sessions != sessions
        or local_qa.binding.listing_axis_hash != canonical_hash(listings)
        or local_qa.binding.method_recipe_hash != seal.method_bound.recipe_hash
        or local_qa.binding.source_rows_hash
        != canonical_hash(local_qa.rows["row_hash"].to_pylist())
    ):
        raise AuthoringError("portfolio_research.local_qa_outcome_binding_mismatch")
    panel_resolver = ArtifactResolver(artifact_root)
    panel_ref = panel_resolver.feature_panel_manifest_uri(source.panel_snapshot_hash)
    available = FeaturePanelReader(panel_resolver).available_sessions(panel_ref)
    # Include the owner-derived entry/holding-end sessions, not a weekday guess.
    available = tuple(
        sorted(
            set(available)
            | {v["entry_session"] for v in schedule}
            | {v["holding_end_session"] for v in schedule}
        )
    )
    epoch = canonical_hash(
        {
            "kind": "ResearchPortfolioMarketEpoch",
            "universe": source.universe_revision,
            "listing_axis": listings,
            "panel": source.panel_snapshot_hash,
        }
    )
    inputs = PortfolioTradabilityMarketInputs(
        source.input_binding_hash,
        epoch,
        manifest.profile.market_profile_id,
        listings,
        source.universe_revision,
        sessions,
        available,
    )
    builder = HistoricalTradabilityBuilder(
        market_store=market, artifact_root=output, capacity=capacity
    )
    try:
        published = builder.build(market_inputs=inputs, should_cancel=cancellation)
    except TradabilityBuildCancelled as error:
        # The builder stops between immutable chunks, the same safe boundary
        # the authored experiment reports to Task Control as cancellation.
        raise PortfolioExperimentCancelled(
            "portfolio_research.market_preparation_cancelled"
        ) from error
    decision, execution, adv = read_tradability_matrices(
        store=builder.artifacts,
        decision=published.decision,
        execution=published.execution,
        sessions=sessions,
        listings=listings,
    )
    membership = market.membership_schedule(
        manifest.profile.market_profile_id,
        sessions=sessions,
        fallback_listing_ids=tuple(item.listing_id for item in manifest.listings),
    )
    decision = decision.copy()
    for index, session in enumerate(sessions):
        members = set(membership.members(session))
        decision[index] &= np.asarray([listing in members for listing in listings], dtype=bool)
    decision.setflags(write=False)
    realized = outcome_returns(
        reader.read_development_sessions(reader.manifest_uri(outcome.snapshot_hash), sessions)
        if local_qa is None
        else local_qa.rows,
        sessions=sessions,
        listings=listings,
    )
    if (
        local_qa is not None
        and sha256(np.ascontiguousarray(realized, dtype="<f8").tobytes()).hexdigest()
        != local_qa.binding.returns_hash
    ):
        raise AuthoringError("portfolio_research.local_qa_outcome_values_mismatch")
    classification, _coverage_hash = _sector_by_listing_id(
        workspace=source_workspace,
        market_profile_id=manifest.profile.market_profile_id,
        panel_manifest=panel_resolver.load_feature_panel_manifest(panel_ref),
        listing_ids=listings,
    )
    sectors, sector_anchor = sector_exposure_lanes(
        classification, listings=listings, sessions=sessions
    )
    mark_store = SessionMarkArtifactStore(output, capacity=capacity)
    marks = publish_market_session_marks(
        market=market,
        store=mark_store,
        manifest=manifest,
        listing_ids=listings,
        sessions=sessions,
        source_watermark_hash=published.decision.source_watermark_hash,
        corporate_action_identity=outcome.corporate_action_identity,
        availability=resolve_session_mark_availability(),
    )
    transition = resolve_portfolio_state_transition_binding(
        surface=marks,
        rebalance_clock=source.clock.binding,
        playpen_root=resolve_playpen_root(Path(__file__)),
        execution_method_binding_hash=seal.method_bound.binding_hash if local_qa is None else None,
    )
    reference = resolve_reference_mark_lane(
        store=mark_store,
        surface=marks,
        binding=transition,
        entry_session_by_formation={v["formation_session"]: v["entry_session"] for v in schedule},
        formation_sessions=sessions,
        ordered_listing_ids=listings,
    )
    decision, exclusions = resolve_research_universe(
        spec=spec, sessions=sessions, listings=listings, decision=decision, realized=realized
    )
    if exclusions:
        excluded = {v.listing_id for v in exclusions}
        sector_anchor = np.mean(sectors[..., [v not in excluded for v in listings]], axis=-1)
        sector_anchor.setflags(write=False)
    matrix: npt.NDArray[np.float64] = np.full(
        (len(sessions), len(listings)), np.nan, dtype=np.float64
    )
    eligibility: npt.NDArray[np.bool_] = np.zeros(matrix.shape, dtype=np.bool_)
    positions = {s: i for i, s in enumerate(sessions)}
    for i, session in enumerate(scores.sessions):
        j = positions[session]
        matrix[j] = scores.scores[i]
        eligibility[j] = decision[j] & scores.available[i]
    matrix.setflags(write=False)
    eligibility.setflags(write=False)
    # The declared observed-data quarantine applies to both the opportunity set
    # and benchmark. Within that population, never filter members by daily outcomes.
    passive = {s: realized[i][decision[i]] for i, s in enumerate(sessions)}
    labels = {
        item.listing_id: item.symbol
        for item in market.listing_scope(manifest, listing_ids=listings)
    }
    return PortfolioExperimentInputs(
        sessions,
        listings,
        matrix,
        eligibility,
        execution,
        realized,
        adv,
        sectors,
        sector_anchor,
        passive,
        reference.lane,
        {
            "tradability": published.bundle.bundle_hash,
            "outcome": outcome.snapshot_hash if local_qa is None else local_qa.binding.content_hash,
            "session_mark": marks.surface_hash,
            "transition": transition.binding_hash,
            "source": source.input_binding_hash,
            **(
                {"data_quarantine": canonical_hash([v.model_dump(mode="json") for v in exclusions])}
                if exclusions
                else {}
            ),
        },
        listing_labels=tuple(labels[listing] for listing in listings),
        data_exclusions=exclusions,
        market_decision_eligible=decision,
        transition_binding=transition,
        tradability_decision_hash=published.decision.surface_hash,
    )
