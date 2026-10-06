"""Run one authored research experiment through the Host authoring boundary.

Commands:

    validate  parse the document and prove the Desk kind and capability install
    freeze    resolve the authored handles against a real workspace and seal a Program
    run       seal, execute for real, and write immutable development evidence
    replay    seal again, prove the identity is unchanged, and read the evidence back

`run` and `replay` are genuinely different calls. They used to resolve to the
same dispatcher call as `freeze`, so nothing ever executed and nothing was ever
replayed.

Human, installed Agent, and external automation all enter here; `--actor-kind`
is provenance only and never changes routing or numbers.

Exit codes: 0 ok, 2 authoring, 3 authority, 4 execution, 5 replay mismatch.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

PLAYPEN_ROOT = Path(__file__).resolve().parents[1]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"
if str(PLAYPEN_SRC) not in sys.path:
    sys.path.insert(0, str(PLAYPEN_SRC))

# BLAS/OpenMP must be bounded before importing NumPy-bearing Desk modules; only when run as a
# script, so a test importing this module keeps its own threads and network (W11).
if __name__ == "__main__":
    for _thread_variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[_thread_variable] = "1"
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"

from alphalattice.control.observation_runtime.telemetry.process_metrics import (
    ResearchRuntimeRequest,
    RuntimeWorkload,
    bind_process_logical_processors,
    resolve_runtime_capacity_plan,
    runtime_machine_capacity,
)
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_experiment_dispatcher,
    build_research_program_workflow,
    host_resolved_factor_inventory,
)
from alphalattice.control.product_host.research_authoring.authority import (
    InstalledResearchSnapshot,
)
from alphalattice.control.product_host.research_authoring.execution import (
    PanelMethodologyRuntimeRoots,
    build_installed_desk_executors,
    build_panel_methodology_authority_resolver,
)
from alphalattice.control.product_host.research_authoring.panel_methodology_context import (
    PANEL_METHODOLOGY_LOCATION_IDS,
    discover_panel_methodology_context,
    load_panel_methodology_context_manifest,
    render_panel_methodology_context,
)
from alphalattice.control.product_host.research_authoring.panel_methodology_sources import (
    PanelMethodologySourceRoots,
)
from alphalattice.control.research_program.authoring.document import load_authoring_document
from alphalattice.control.research_program.authoring.workflow import ResearchProgramStore
from alphalattice.control.workspace_runtime.reader_threads import apply_reader_threads
from alphalattice.foundation.factor_research.experiments.authoring import FACTOR_EXPERIMENT_KIND
from alphalattice.investment.alpha_research.experiments.authoring import (
    ALPHA_EXPERIMENT_KIND,
    AlphaExperimentCompiler,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_authoring import (
    PANEL_ALPHA_FOLD_RSS_FUSE_BYTES,
    PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES,
    AlphaResearchMethodologyCatalog,
    PanelResearchMethodologyRequest,
    PanelResearchRuntimeCaps,
    methodology_section,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_models import (
    build_panel_model_catalog,
)
from alphalattice.investment.alpha_research.experiments.policies import load_alpha_split_policy
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    INSTALLED_PANEL_VIEW_IDS,
)
from alphalattice.investment.alpha_research.scores.score_filters import (
    build_installed_alpha_score_filter_catalog,
)
from alphalattice.investment.alpha_research.targets.total_return import (
    TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID,
)
from alphalattice.investment.portfolio_strategy_lab.policies.catalog import (
    build_installed_portfolio_policy_catalog,
)
from alphalattice.investment.portfolio_strategy_lab.research_loop import (
    panel_methodology_execution,
)
from alphalattice.investment.risk_research.estimators.catalog import (
    build_installed_risk_estimator_catalog,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind, seal_actor_submission
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskExperimentExecutor,
    ResearchExperimentEnvelope,
)

_AUTHORITY_FAILURES = frozenset(
    {
        "research_authoring.universe_handle_unresolved",
        "research_authoring.snapshot_handle_unresolved",
        "research_authoring.no_sessions_resolved",
    }
)

_FEATURE_OBSERVATION_CLOCK_SUPERSEDED = "SUPERSEDED_BY_FEATURE_OBSERVATION_CLOCK"
_FEATURE_OBSERVATION_CLOCK_INPUT = "E2F43C4_STABLE_STAGE2_INPUTS"
_SCIENTIFIC_NUMERICAL_LIFECYCLE_COMMANDS = frozenset(
    {
        "preflight",
        "run",
        "resume",
        "alpha-run",
        "alpha-resume",
        "score-filter",
        "portfolio-run",
        "portfolio-resume",
    }
)


class _AlphaCompilerExecutor(Protocol):
    @property
    def compiler(self) -> AlphaExperimentCompiler: ...


def _admit_panel_input_lifecycle(document: dict[str, object], *, command: str) -> str | None:
    """Keep a superseded input readable while refusing new numerical publication.

    The Feature owner confirmed that this historical surface labels mostly
    close(T-1) observations as current for a close(T) information clock.  This
    composition-root admission does not reinterpret or patch FactorSpec.  It
    permits immutable inspect/replay and blocks the commands that could seal or
    publish more scientific evidence from the stale surface.
    """

    section = methodology_section(document)
    if section is None or section.get("input_method_id") != _FEATURE_OBSERVATION_CLOCK_INPUT:
        return None
    if command in _SCIENTIFIC_NUMERICAL_LIFECYCLE_COMMANDS:
        raise AuthoringError("alpha_research.feature_observation_clock_not_admitted")
    return _FEATURE_OBSERVATION_CLOCK_SUPERSEDED


def _admit_panel_stage_command(document: dict[str, object], *, command: str) -> None:
    """Bind explicit lifecycle verbs to the stage sealed in typed authoring."""

    section = methodology_section(document)
    if section is None or command in {
        "validate",
        "freeze",
        "preflight",
        "inspect",
        "replay",
    }:
        return
    request = PanelResearchMethodologyRequest.from_mapping(section)
    expected = {
        "ALPHA_ONLY": {"alpha-run", "alpha-resume"},
        "SCORE_FILTER_ONLY": {"score-filter"},
        "PORTFOLIO_ONLY": {"portfolio-run", "portfolio-resume"},
        "END_TO_END": {"run", "resume"},
    }[request.execution_stage]
    if command not in expected:
        raise AuthoringError("alpha_research.panel_methodology_stage_command_mismatch")


_REPLAY_FAILURES = frozenset(
    {
        "research_authoring.replay_identity_mismatch",
        "research_authoring.evidence_unavailable_for_replay",
        "research_authoring.evidence_identity_conflict",
    }
)


def _pin_numerical_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["ALPHALATTICE_NETWORK_DISABLED"] = "1"


def _exit_code(error: AuthoringError) -> int:
    """Distinguish why a request was refused, so a caller can act on it."""

    message = str(error)
    if message in _AUTHORITY_FAILURES:
        return 3
    if message in _REPLAY_FAILURES:
        return 5
    if message == "research_authoring.desk_executor_not_installed":
        return 4
    return 2


def _program_report(program: Any) -> dict[str, Any]:
    """Render a sealed Program without dumping thousands of resolved sessions."""

    sessions = tuple(program.resolved_sessions)
    return {
        "kind": program.kind,
        "program_hash": program.program_hash,
        "desk_program_hash": program.desk_program_hash,
        "catalog_hash": program.catalog_hash,
        "method_binding_hash": program.method_binding_hash,
        "parameter_domain_hash": program.parameter_domain_hash,
        "authority_hash": program.authority_hash,
        "resolved_session_count": len(sessions),
        "first_resolved_session": str(sessions[0]) if sessions else None,
        "last_resolved_session": str(sessions[-1]) if sessions else None,
    }


def _call_counts(value: int | None) -> dict[str, Any]:
    """The four numerical-call counters, which every command reports together.

    `0` is a claim that no numerical owner was called, and only a path that
    resolves no executor may make it. `None` says this command does not answer
    for those counters at all. Splitting them across four literals at four call
    sites is how one of them once said `0` on a path that had run.
    """

    return {
        "fit_call_count": value,
        "predict_call_count": value,
        "metric_call_count": value,
        "solver_call_count": value,
    }


def _evidence_report(
    command: str,
    evidence: Any,
    binding: Any,
    *,
    numerical_call_count: int,
    scientific_disposition: object,
) -> dict[str, Any]:
    """The body every command that ends in evidence publishes.

    One shape for the sealed readback and for the live lifecycle, because a
    reader comparing a `replay` against the `run` it replays must not have to
    reconcile two spellings of the same answer. Each caller adds only what is
    genuinely its own -- a live-source count, or stage telemetry.
    """

    return {
        "command": command,
        "kind": evidence.kind,
        "program_hash": evidence.program_hash,
        "disposition": evidence.disposition,
        "identity_class": evidence.identity_class,
        "numerical_call_count": numerical_call_count,
        "artifact_uris": list(evidence.artifact_uris),
        "formation_session_count": len(evidence.formation_sessions),
        "evidence_hash": evidence.evidence_hash,
        "actor_kind": binding.actor_kind.value,
        "actor_binding_hash": binding.binding_hash,
        "pointer_mutation_count": 0,
        "source_workspace_write_count": 0,
        "scientific_disposition": scientific_disposition,
    }


def _inspect_report(
    program: Any, evidence: Any, binding: Any, *, scientific_disposition: object
) -> dict[str, Any]:
    """The read-only body both inspect routes publish, sealed or live."""

    return {
        "command": "inspect",
        "program": _program_report(program),
        "evidence": (evidence.model_dump(mode="json") if evidence is not None else None),
        "actor_binding_hash": binding.binding_hash,
        "read_only": True,
        "scientific_disposition": scientific_disposition,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_research_experiment", description=__doc__)
    parser.add_argument(
        "positionals",
        nargs="*",
        help="[legacy-command] authoring-document; lifecycle flags are preferred",
    )
    lifecycle = parser.add_mutually_exclusive_group()
    lifecycle.add_argument("--list-methods", dest="command", action="store_const", const="list")
    lifecycle.add_argument("--inspect-method", metavar="METHOD_ID")
    lifecycle.add_argument(
        "--print-context", dest="command", action="store_const", const="print-context"
    )
    lifecycle.add_argument(
        "--preflight",
        dest="command",
        action="store_const",
        const="preflight",
        help="Create or re-check the exact Portfolio-only metadata receipt",
    )
    lifecycle.add_argument("--run", dest="command", action="store_const", const="run")
    lifecycle.add_argument("--resume", dest="command", action="store_const", const="resume")
    lifecycle.add_argument("--alpha-run", dest="command", action="store_const", const="alpha-run")
    lifecycle.add_argument(
        "--alpha-resume", dest="command", action="store_const", const="alpha-resume"
    )
    lifecycle.add_argument(
        "--score-filter", dest="command", action="store_const", const="score-filter"
    )
    lifecycle.add_argument(
        "--portfolio-run",
        dest="command",
        action="store_const",
        const="portfolio-run",
        help="Execute the exact sealed Portfolio Program",
    )
    lifecycle.add_argument(
        "--portfolio-resume",
        dest="command",
        action="store_const",
        const="portfolio-resume",
        help="Resume the exact sealed Portfolio Program or replay completed evidence",
    )
    lifecycle.add_argument("--inspect", dest="command", action="store_const", const="inspect")
    lifecycle.add_argument("--replay", dest="command", action="store_const", const="replay")
    parser.add_argument(
        "--actor-kind",
        choices=tuple(value.value for value in ActorKind),
        default=ActorKind.HUMAN.value,
    )
    parser.add_argument("--actor-id", default="local-researcher")
    parser.add_argument(
        "--research-input", help="Installed local Factor input id in the product workspace"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        required=False,
        help="Workspace whose published Panel and universe the handles resolve against",
    )
    parser.add_argument(
        "--playpen-root",
        type=Path,
        default=PLAYPEN_ROOT,
        help="Root the document's relative output_workspace is resolved against",
    )
    parser.add_argument(
        "--factor-evidence-root",
        type=Path,
        default=None,
        help=(
            "Read-only root holding Factor development evidence. Required by Alpha "
            "experiments, which name a checkpoint handle and never a location."
        ),
    )
    parser.add_argument("--panel-artifact-root", type=Path)
    parser.add_argument(
        # Where the *authority resolver* reads the Panel it verifies. This is a
        # different question from where a Desk executor reads its own inputs,
        # and conflating them is what made naming a Panel location silently
        # redirect the Risk Desk's return-surface reads to a root that holds
        # only Feature artifacts. Defaults to --panel-artifact-root so existing
        # invocations are unchanged.
        "--panel-authority-root",
        type=Path,
    )
    parser.add_argument(
        # A named immutable Panel, for documents that have no methodology
        # section to install one for them. The pair supplies *where to look*;
        # the Feature owner's clock verifier still decides what is there, so a
        # handle naming an unverifiable snapshot refuses exactly as before.
        "--installed-data-snapshot",
        action="append",
        default=[],
        metavar="HANDLE=SNAPSHOT_HASH",
    )
    parser.add_argument("--legacy-panel-artifact-root", type=Path)
    parser.add_argument("--panel-feature-artifact-root", type=Path)
    parser.add_argument("--failed-baseline-workspace", type=Path)
    parser.add_argument("--sector-context-artifact-root", type=Path)
    parser.add_argument("--execution-outcome-artifact-root", type=Path)
    parser.add_argument("--panel-methodology-repository-root", type=Path)
    parser.add_argument("--r0-evidence-root", type=Path)
    parser.add_argument("--r1-evidence-root", type=Path)
    parser.add_argument(
        "--portfolio-market-workspace",
        type=Path,
        help="Read-only local Market Data workspace for Portfolio facts and benchmark",
    )
    parser.add_argument(
        "--portfolio-tradability-artifact-root",
        type=Path,
        help="Read-only artifact root holding the immutable Portfolio tradability bundle",
    )
    parser.add_argument(
        "--risk-return-artifact-root",
        type=Path,
        help="Read-only artifact root holding the Risk input binding's return surface",
    )
    parser.add_argument("--maximum-feature-count", type=int, default=4096)
    parser.add_argument("--maximum-candidate-count", type=int, default=4096)
    parser.add_argument("--maximum-fit-calls", type=int, default=20_000)
    parser.add_argument("--maximum-predict-calls", type=int, default=20_000)
    parser.add_argument("--maximum-metric-calls", type=int, default=100_000)
    parser.add_argument("--maximum-solver-calls", type=int, default=100_000)
    parser.add_argument("--maximum-estimated-wall-seconds", type=int, default=64_800)
    parser.add_argument("--maximum-fold-workers", type=int, default=None)
    parser.add_argument(
        "--program-hash",
        default=None,
        help=(
            "Exact sealed Program: re-checks its Portfolio receipt during --preflight and "
            "is required by --portfolio-run/--portfolio-resume; inspect/replay stay source-free"
        ),
    )
    parser.add_argument(
        "--root-hash",
        default=None,
        help="Exact admitted methodology root for candidate-specific inspect output",
    )
    parser.add_argument(
        "--model-recipe-id",
        default=None,
        help="Exact fixed model recipe to resolve beneath --root-hash",
    )
    parser.add_argument(
        "--context-manifest",
        type=Path,
        default=None,
        help=(
            "Operational routing produced by --print-context. It supplies runtime "
            "locations only; scientific identity always comes from the artifacts a "
            "run actually opens. Explicit --*-root flags win over it."
        ),
    )
    return parser


def _installed_data_snapshots(
    arguments: argparse.Namespace,
) -> tuple[InstalledResearchSnapshot, ...]:
    """The immutable Panel handles an operator named on the command line."""

    values: list[InstalledResearchSnapshot] = []
    for entry in arguments.installed_data_snapshot:
        handle, separator, snapshot = str(entry).partition("=")
        if not separator:
            raise AuthoringError("research_authoring.installed_snapshot_argument_invalid")
        values.append(InstalledResearchSnapshot(handle=handle, panel_snapshot_hash=snapshot))
    return tuple(values)


def _named_panel_locations(arguments: argparse.Namespace) -> dict[str, Path | None]:
    """The runtime locations an operator named on the command line."""

    return {
        "repository_root": arguments.panel_methodology_repository_root,
        "panel_artifact_root": arguments.panel_artifact_root,
        "legacy_panel_artifact_root": arguments.legacy_panel_artifact_root,
        "feature_artifact_root": arguments.panel_feature_artifact_root,
        "failed_baseline_workspace": arguments.failed_baseline_workspace,
        "sector_context_artifact_root": arguments.sector_context_artifact_root,
        "execution_outcome_artifact_root": arguments.execution_outcome_artifact_root,
        "r0_evidence_root": arguments.r0_evidence_root,
        "r1_evidence_root": arguments.r1_evidence_root,
        "portfolio_market_workspace": arguments.portfolio_market_workspace,
        "portfolio_tradability_artifact_root": (arguments.portfolio_tradability_artifact_root),
        "risk_return_artifact_root": arguments.risk_return_artifact_root,
    }


def _panel_runtime_workload(
    *, request: PanelResearchMethodologyRequest, command: str
) -> RuntimeWorkload:
    """Describe the actual owner work this invocation may open."""

    if command in {"validate", "freeze", "preflight", "inspect", "replay"}:
        return "METADATA_PREFLIGHT"
    return cast(
        RuntimeWorkload,
        {
            "ALPHA_ONLY": "ALPHA_FOLDS",
            "SCORE_FILTER_ONLY": "SCORE_FILTER_NUMERICAL",
            "PORTFOLIO_ONLY": "PORTFOLIO_NUMERICAL",
            "END_TO_END": "ALPHA_FOLDS",
        }[request.execution_stage],
    )


def _panel_runtime(
    arguments: argparse.Namespace,
    document: dict[str, object],
    *,
    command: str,
) -> tuple[PanelMethodologyRuntimeRoots, PanelResearchRuntimeCaps]:
    named = _named_panel_locations(arguments)
    if arguments.context_manifest is not None:
        routed = load_panel_methodology_context_manifest(arguments.context_manifest)
        # A manifest routes; it never overrides an operator who named a location.
        named = {
            key: (value if value is not None else routed.get(key)) for key, value in named.items()
        }
    section = methodology_section(document)
    if section is None:
        raise AuthoringError("alpha_research.methodology_not_selected")
    request = PanelResearchMethodologyRequest.from_mapping(section)
    workload = _panel_runtime_workload(request=request, command=command)
    required = {
        "repository_root",
        "panel_artifact_root",
        "legacy_panel_artifact_root",
        "feature_artifact_root",
        "failed_baseline_workspace",
        "sector_context_artifact_root",
        "execution_outcome_artifact_root",
    }
    if request.execution_stage in {"PORTFOLIO_ONLY", "END_TO_END"}:
        required.update(
            {
                "portfolio_market_workspace",
                "portfolio_tradability_artifact_root",
                "r0_evidence_root",
                "risk_return_artifact_root",
            }
        )
        if request.risk_method_ids == ("R0", "R1"):
            required.add("r1_evidence_root")
    missing = tuple(key for key in sorted(required) if named[key] is None)
    if missing:
        raise AuthoringError(
            "alpha_research.panel_methodology_runtime_root_missing:" + ",".join(missing)
        )
    runtime_section = document.get("runtime", {})
    if not isinstance(runtime_section, dict):
        raise AuthoringError("research_authoring.runtime_profile_invalid")
    try:
        runtime_request = ResearchRuntimeRequest.model_validate(runtime_section)
        if arguments.maximum_fold_workers is not None:
            runtime_request = runtime_request.model_copy(
                update={
                    "parallelism": runtime_request.parallelism.model_copy(
                        update={"fold_workers": arguments.maximum_fold_workers}
                    )
                }
            )
        machine = runtime_machine_capacity()
        runtime_plan = resolve_runtime_capacity_plan(
            request=runtime_request,
            machine=machine,
            fold_count=5,
            parent_reservation_bytes=PANEL_ALPHA_PARENT_RSS_RESERVATION_BYTES,
            worker_reservation_bytes=PANEL_ALPHA_FOLD_RSS_FUSE_BYTES,
            workload=workload,
        )
    except (ValueError, RuntimeError) as error:
        raise AuthoringError("research_authoring.runtime_profile_invalid") from error
    if runtime_plan.admission != "ADMITTED":
        raise AuthoringError(runtime_plan.refusal_code or "research_authoring.runtime_not_admitted")
    try:
        capacity_receipt = bind_process_logical_processors(
            runtime_plan.process_logical_processor_limit,
            library_thread_limit=runtime_plan.blas_threads,
        )
        runtime_plan = runtime_plan.with_applied_logical_processors(
            tuple(
                int(value)
                for value in cast(list[int], capacity_receipt["applied_logical_processors"])
            )
        )
    except RuntimeError as error:
        raise AuthoringError("research_authoring.runtime_capacity_unenforceable") from error
    # The admitted DuckDB count, applied through the budget's one mechanism (B7), so each DuckDB
    # connection this run opens takes it instead of DuckDB's own default (V5).
    apply_reader_threads(runtime_plan.duckdb_threads)
    roots = PanelMethodologyRuntimeRoots(
        source=PanelMethodologySourceRoots(
            repository_root=cast(Path, named["repository_root"]),
            panel_artifact_root=cast(Path, named["panel_artifact_root"]),
            legacy_panel_artifact_root=cast(Path, named["legacy_panel_artifact_root"]),
            feature_artifact_root=cast(Path, named["feature_artifact_root"]),
            failed_baseline_workspace=cast(Path, named["failed_baseline_workspace"]),
            sector_context_artifact_root=cast(Path, named["sector_context_artifact_root"]),
            execution_outcome_artifact_root=cast(Path, named["execution_outcome_artifact_root"]),
        ),
        r0_evidence_root=named["r0_evidence_root"],
        r1_evidence_root=named["r1_evidence_root"],
        portfolio_market_workspace=named["portfolio_market_workspace"],
        portfolio_tradability_artifact_root=named["portfolio_tradability_artifact_root"],
        risk_return_artifact_root=named["risk_return_artifact_root"],
    )
    caps = PanelResearchRuntimeCaps(
        maximum_feature_count=arguments.maximum_feature_count,
        maximum_candidate_count=arguments.maximum_candidate_count,
        maximum_fit_calls=arguments.maximum_fit_calls,
        maximum_predict_calls=arguments.maximum_predict_calls,
        maximum_metric_calls=arguments.maximum_metric_calls,
        maximum_solver_calls=arguments.maximum_solver_calls,
        maximum_estimated_wall_seconds=arguments.maximum_estimated_wall_seconds,
        maximum_estimated_peak_memory_bytes=runtime_plan.admitted_memory_budget_bytes,
        maximum_fold_workers=runtime_plan.fold_workers,
        runtime_plan_hash=runtime_plan.plan_hash,
        runtime_workload=runtime_plan.workload,
        runtime_profile=runtime_plan.profile,
        runtime_profile_resolution=runtime_plan.profile_resolution,
        process_logical_processor_limit=runtime_plan.process_logical_processor_limit,
        applied_logical_processor_ids=runtime_plan.applied_logical_processor_ids,
        duckdb_threads=runtime_plan.duckdb_threads,
        blas_threads=runtime_plan.blas_threads,
        lightgbm_threads_per_fit=runtime_plan.lightgbm_threads_per_fit,
    )
    return roots, caps


def _installed_method_descriptors() -> tuple[dict[str, object], ...]:
    methodology = AlphaResearchMethodologyCatalog()
    model = build_panel_model_catalog()
    risk = build_installed_risk_estimator_catalog()
    portfolio = build_installed_portfolio_policy_catalog()
    score_filters = build_installed_alpha_score_filter_catalog()
    values: list[dict[str, object]] = [
        {
            "owner": "factor_research.experiments",
            "method_id": FACTOR_EXPERIMENT_KIND,
            "method_kind": "FACTOR",
        },
        {
            "owner": "alpha_research.targets",
            "method_id": TOTAL_RETURN_ALPHA_TARGET_RECIPE_ID,
            "method_kind": "TARGET",
        },
    ]
    values.extend(
        {
            "owner": value.owner,
            "method_id": value.method_id,
            "method_kind": "SCORE_FILTER",
            "descriptor_hash": value.descriptor_hash,
            "formula_id": value.formula_id,
            "admitted_span_domain": list(value.admitted_span_domain),
            "admitted_input_modes": list(value.admitted_input_modes),
            "row_axis_policy": value.row_axis_policy,
            "absence_policy": value.absence_policy,
        }
        for value in score_filters.descriptors
    )
    values.extend(
        {
            "owner": "alpha_modeling.adapters",
            "method_id": family_id,
            "method_kind": "MODEL",
            "adapter_id": adapter_id,
            "numerical_binding_hash": next(
                value.numerical_binding_hash
                for value in model.binding.ordered_capabilities
                if value.adapter_id == adapter_id
            ),
        }
        for family_id, adapter_id in (
            ("RIDGE", "dynamic_panel_regularized_linear"),
            ("ELASTIC_NET", "dynamic_panel_regularized_linear"),
            ("LIGHTGBM", "dynamic_panel_lightgbm"),
        )
    )
    values.extend(
        {
            "owner": value.owner,
            "method_id": value.method_id,
            "method_kind": "RESEARCH_METHODOLOGY",
            "descriptor_hash": value.descriptor_hash,
            "description": value.description,
            "lifecycle_commands": list(value.lifecycle_commands),
        }
        for value in methodology.descriptors
    )
    values.extend(
        {
            "owner": "alpha_research.inputs",
            "method_id": method_id,
            "method_kind": "FEATURE_VIEW",
            "descriptor_resolution": "AXIS_BOUND_DURING_PREFLIGHT",
        }
        for method_id in INSTALLED_PANEL_VIEW_IDS
    )
    values.extend(
        {
            "owner": "risk_research.estimators",
            "method_id": value.recipe_schema_id,
            "adapter_id": value.adapter_id,
            "method_kind": "RISK",
            "numerical_binding_hash": value.numerical_binding_hash,
        }
        for value in risk.binding.ordered_capabilities
    )
    values.extend(
        {
            "owner": "portfolio_strategy_lab.policies",
            "method_id": value.policy_id,
            "method_kind": "PORTFOLIO",
            "solver_backed": value.solver_backed,
        }
        for value in portfolio.binding.ordered_capabilities
    )
    return tuple(
        sorted(values, key=lambda value: (str(value["method_kind"]), str(value["method_id"])))
    )


def _selected_envelope(document: dict[str, object]) -> ResearchExperimentEnvelope:
    """Read the document's own request, before any Host exists.

    The envelope is caller input and carries no authority, but it does name the
    Desk kind -- and that is enough to install exactly the compiler and executor
    this run needs instead of every Desk's. Building all of them was what made a
    Factor run depend on Risk having published a return surface.
    """

    section = document.get("experiment")
    if not isinstance(section, dict):
        raise AuthoringError("research_authoring.experiment_section_missing")
    return ResearchExperimentEnvelope.create(**section)


def _installed_executors(
    arguments: argparse.Namespace,
    document: dict[str, object],
    envelope: ResearchExperimentEnvelope,
    panel_runtime: tuple[PanelMethodologyRuntimeRoots, PanelResearchRuntimeCaps] | None = None,
    *,
    preflight_metadata_only: bool = False,
) -> tuple[DeskExperimentExecutor, ...]:
    roots, caps = panel_runtime if panel_runtime is not None else (None, None)
    executors: tuple[DeskExperimentExecutor, ...] = build_installed_desk_executors(
        envelope=envelope,
        workspace=arguments.workspace,
        workspace_root=arguments.playpen_root,
        document=document,
        factor_evidence_root=arguments.factor_evidence_root,
        alpha_split_policy=load_alpha_split_policy(),
        artifact_root=arguments.panel_artifact_root,
        panel_methodology_roots=roots,
        panel_runtime_caps=caps,
        sealed_preflight_program_hash=arguments.program_hash,
        preflight_metadata_only=preflight_metadata_only,
    )
    return executors


def main(argv: tuple[str, ...] | None = None) -> int:
    _pin_numerical_threads()
    command_started = perf_counter()
    arguments = _build_parser().parse_args(argv)
    report: dict[str, object]
    try:
        positions = tuple(arguments.positionals)
        command = arguments.command
        document_path: Path | None = None
        if positions and positions[0] in {"validate", "freeze", "run", "replay"}:
            if command is not None:
                raise AuthoringError("research_authoring.lifecycle_command_duplicated")
            command = positions[0]
            positions = positions[1:]
        if arguments.inspect_method is not None:
            command = "inspect-method"
        if command is None:
            raise AuthoringError("research_authoring.lifecycle_command_missing")
        if command == "list":
            if positions:
                raise AuthoringError("research_authoring.document_not_admitted_for_method_list")
            report = {
                "command": "list-methods",
                "methods": list(_installed_method_descriptors()),
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if command == "print-context":
            if positions:
                raise AuthoringError("research_authoring.document_not_admitted_for_context")
            search = tuple(
                value
                for value in (arguments.workspace, arguments.playpen_root)
                if value is not None
            )
            # An operator who already knows where an externally published
            # artifact lives supplies the location here rather than by hand
            # afterwards. It is still decided at its owner, and it still appears
            # in the report as a place that was looked at.
            named = {
                key: value
                for key, value in _named_panel_locations(arguments).items()
                if key in PANEL_METHODOLOGY_LOCATION_IDS and value is not None
            }
            if not search and not named:
                raise AuthoringError("research_authoring.workspace_required")
            report = render_panel_methodology_context(
                discover_panel_methodology_context(search_roots=search, named_locations=named),
                searched_roots=tuple(
                    dict.fromkeys(search + tuple(named[key] for key in sorted(named)))
                ),
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["refusal"] is None else 2
        if command == "inspect-method":
            if positions:
                raise AuthoringError("research_authoring.document_not_admitted_for_method_inspect")
            matches = tuple(
                value
                for value in _installed_method_descriptors()
                if value["method_id"] == arguments.inspect_method
            )
            if not matches:
                raise AuthoringError("research_authoring.method_not_installed")
            report = {"command": "inspect-method", "methods": list(matches)}
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if len(positions) != 1:
            raise AuthoringError("research_authoring.document_path_required")
        document_path = Path(positions[0])
        if arguments.workspace is None:
            raise AuthoringError("research_authoring.workspace_required")
        document = load_authoring_document(document_path.read_text(encoding="utf-8"))
        bound_factor_input = None
        if arguments.research_input is not None:
            from alphalattice.control.product_host.composition.research_workspace import (
                read_research_workspace_manifest,
            )
            from alphalattice.control.product_host.research_authoring.factor_inputs import (
                factor_workflow,
                normalize_factor_document,
                read_factor_bundle,
            )

            if ActorKind(arguments.actor_kind) is ActorKind.INSTALLED_AGENT:
                raise AuthoringError("research_authoring.actor_submission_provenance_invalid")
            bindings = [
                v
                for v in read_research_workspace_manifest(arguments.workspace).experiment_inputs
                or ()
                if v.input_id == arguments.research_input
            ]
            if len(bindings) != 1:
                raise AuthoringError("research_experiment.input_not_admitted")
            bound_factor_input = bindings[0]
            bundle = read_factor_bundle(arguments.workspace, bound_factor_input.binding_hash)
            document = normalize_factor_document(document, bound_factor_input, bundle)
        selected = _selected_envelope(document)
        _admit_panel_stage_command(document, command=command)
        scientific_disposition = _admit_panel_input_lifecycle(document, command=command)
        actor_kind = ActorKind(arguments.actor_kind)
        try:
            # The CLI has no Agent-execution receipt input. Ask the actor owner
            # before any Desk source/Feature resolver opens; the real Program
            # binding is sealed later over the compiled Program hash.
            seal_actor_submission(
                actor_kind=actor_kind,
                actor_id=arguments.actor_id,
                submission_hash=selected.envelope_hash,
            )
        except ValueError as error:
            raise AuthoringError(
                "research_authoring.actor_submission_provenance_invalid"
            ) from error
        selected_methodology = methodology_section(document)
        if bound_factor_input is not None:
            if selected.kind != FACTOR_EXPERIMENT_KIND:
                raise AuthoringError("research_experiment.input_not_admitted")
            workflow, _authority, execution_preview = factor_workflow(
                workspace=arguments.workspace,
                binding_hash=bound_factor_input.binding_hash,
                document=document,
            )
            actor = {"actor_kind": actor_kind, "actor_id": arguments.actor_id}
            if command in {"validate", "freeze", "preflight"}:
                sealed = workflow.prepare(document, **actor)
                if command == "preflight":
                    workflow.preflight(document, **actor)
                report = {
                    "command": command,
                    "program": _program_report(sealed.program),
                    "document": document,
                    "execution_preview": execution_preview,
                    **_call_counts(0),
                }
            elif command == "inspect":
                program, evidence, actor_binding, standing = workflow.inspect(document, **actor)
                report = {
                    **_inspect_report(
                        program,
                        evidence,
                        actor_binding,
                        scientific_disposition=scientific_disposition,
                    ),
                    "standing": standing,
                }
            elif command in {"run", "resume", "replay"}:
                action = {
                    "run": workflow.run_sealed,
                    "resume": workflow.resume,
                    "replay": workflow.replay,
                }[command]
                evidence, actor_binding = action(document, **actor)
                report = _evidence_report(
                    command,
                    evidence,
                    actor_binding,
                    numerical_call_count=evidence.numerical_call_count,
                    scientific_disposition=scientific_disposition,
                )
            else:
                raise AuthoringError("research_experiment.command_not_admitted")
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        if command in {"portfolio-run", "portfolio-resume"} and arguments.program_hash is None:
            request = (
                PanelResearchMethodologyRequest.from_mapping(selected_methodology)
                if selected_methodology is not None
                else None
            )
            if request is not None and request.execution_stage == "PORTFOLIO_ONLY":
                raise AuthoringError("alpha_research.sealed_portfolio_preflight_required")
        sealed_readback = arguments.program_hash is not None and command in {
            "inspect",
            "replay",
            "resume",
            "alpha-resume",
        }
        if sealed_readback:
            workflow = build_research_program_workflow(
                workspace=arguments.workspace,
                workspace_root=arguments.playpen_root,
                executors=(),
            )
            actor = {
                "actor_kind": actor_kind,
                "actor_id": arguments.actor_id,
            }
            if command == "inspect":
                program, evidence, binding = workflow.inspect_sealed(
                    document,
                    program_hash=arguments.program_hash,
                    **actor,
                )
                candidates = None
                if arguments.root_hash is not None or arguments.model_recipe_id is not None:
                    if arguments.root_hash is None or arguments.model_recipe_id is None:
                        raise AuthoringError("alpha_research.fixed_candidate_handle_incomplete")
                    resolver = panel_methodology_execution.resolve_fixed_candidate_score_surfaces
                    candidates = [
                        value.model_dump(mode="json")
                        for value in resolver(
                            output_workspace=(
                                arguments.playpen_root / selected.output_workspace
                            ).resolve(),
                            root_hash=arguments.root_hash,
                            model_recipe_id=arguments.model_recipe_id,
                        )
                    ]
                report = {
                    **_inspect_report(
                        program,
                        evidence,
                        binding,
                        scientific_disposition=scientific_disposition,
                    ),
                    "fixed_candidate_score_surfaces": candidates,
                    "live_source_resolution_count": 0,
                    **_call_counts(0),
                }
            else:
                evidence, binding = workflow.readback_sealed(
                    document,
                    program_hash=arguments.program_hash,
                    **actor,
                )
                report = {
                    **_evidence_report(
                        command,
                        evidence,
                        binding,
                        numerical_call_count=0,
                        scientific_disposition=scientific_disposition,
                    ),
                    "live_source_resolution_count": 0,
                    **_call_counts(0),
                }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        # Each Desk resolves its own authority and no other Desk's.
        #
        # The Factor inventory used to be resolved for every kind, before the
        # branch. Since it reads per-factor methodology identity off the active
        # Panel, a workspace whose Panel predates that field could not run a
        # *Risk* experiment -- a Desk that reads no Factor axis at all. Installing
        # one Desk must never require another Desk's runtime authority to resolve.
        factor_inventory: tuple[Any, ...] = ()
        executors: tuple[Any, ...] = ()
        alpha_compiler = None
        authority = None
        panel_runtime = None
        methodology = methodology_section(document)
        preflight_metadata_only = (
            command == "preflight"
            and methodology is not None
            and methodology.get("execution_stage") == "PORTFOLIO_ONLY"
        )
        if methodology_section(document) is not None:
            panel_runtime = _panel_runtime(arguments, document, command=command)
            authority = build_panel_methodology_authority_resolver(
                workspace=arguments.workspace,
                roots=panel_runtime[0],
                data_snapshot_handle=selected.data_snapshot_handle,
            )
        if selected.kind == FACTOR_EXPERIMENT_KIND:
            factor_inventory = host_resolved_factor_inventory(
                workspace=arguments.workspace,
                market_profile_id=selected.universe_handle,
            )
        elif selected.kind == ALPHA_EXPERIMENT_KIND:
            # Alpha's compiler needs installed catalogs and a Factor development
            # receipt to admit a document at all, and its executor is what
            # resolves them. Taking the compiler off the executor is deliberate:
            # a separately-built one could be constructed against different
            # artifacts, and the run would then fail its own identity check with
            # nothing to explain why.
            executors = _installed_executors(
                arguments,
                document,
                selected,
                panel_runtime=panel_runtime,
                preflight_metadata_only=preflight_metadata_only,
            )
            alpha_compiler = cast(_AlphaCompilerExecutor, executors[0]).compiler
        if command in {"validate", "freeze"}:
            dispatcher = build_research_experiment_dispatcher(
                workspace=arguments.workspace,
                artifact_root=(arguments.panel_authority_root or arguments.panel_artifact_root),
                installed_snapshots=_installed_data_snapshots(arguments),
                authority=authority,
                factor_inventory=factor_inventory,
                alpha_compiler=alpha_compiler,
            )
            if command == "validate":
                envelope = dispatcher.validate(document)
                report = {
                    "command": "validate",
                    "kind": envelope.kind,
                    "envelope_hash": envelope.envelope_hash,
                    "installed_kinds": list(dispatcher.kinds),
                }
            else:
                program, binding = dispatcher.freeze(
                    document,
                    actor_kind=actor_kind,
                    actor_id=arguments.actor_id,
                )
                report = {
                    "command": "freeze",
                    "kind": program.kind,
                    "program_hash": program.program_hash,
                    "desk_program_hash": program.desk_program_hash,
                    "catalog_hash": program.catalog_hash,
                    "method_binding_hash": program.method_binding_hash,
                    "parameter_domain_hash": program.parameter_domain_hash,
                    "authority_hash": program.authority_hash,
                    "resolved_session_count": len(program.resolved_sessions),
                    "actor_kind": binding.actor_kind.value,
                    "actor_binding_hash": binding.binding_hash,
                }
        else:
            # Executors read workspace inputs the composition root owns. This
            # never takes a writer lease on the source workspace: an unpublished
            # return surface is reported, not published on the caller's behalf.
            if not executors:
                executors = _installed_executors(
                    arguments,
                    document,
                    selected,
                    panel_runtime=panel_runtime,
                    preflight_metadata_only=preflight_metadata_only,
                )
            workflow = build_research_program_workflow(
                workspace=arguments.workspace,
                artifact_root=(arguments.panel_authority_root or arguments.panel_artifact_root),
                installed_snapshots=_installed_data_snapshots(arguments),
                workspace_root=arguments.playpen_root,
                executors=executors,
                authority=authority,
                factor_inventory=factor_inventory,
                alpha_compiler=alpha_compiler,
            )
            actor = {
                "actor_kind": actor_kind,
                "actor_id": arguments.actor_id,
            }
            if command == "preflight":
                output_workspace = (arguments.playpen_root / selected.output_workspace).resolve()
                if arguments.program_hash is None:
                    program, binding = workflow.preflight(document, **actor)
                else:
                    sealed_program = ResearchProgramStore(output_workspace).load(
                        program_hash=arguments.program_hash,
                        document=document,
                    )
                    if sealed_program is None:
                        raise AuthoringError("research_authoring.sealed_program_required")
                    program = sealed_program
                    binding = seal_actor_submission(
                        actor_kind=actor_kind,
                        actor_id=arguments.actor_id,
                        submission_hash=program.program_hash,
                    )
                receipt_publisher = getattr(executors[0], "seal_preflight_receipt", None)
                receipt = (
                    receipt_publisher(
                        program=program,
                        output_workspace=output_workspace,
                    )
                    if receipt_publisher is not None
                    else None
                )
                executor_preflight = getattr(executors[0], "preflight", None)
                feature_preflight = getattr(
                    getattr(getattr(executors[0], "inputs", None), "plan", None),
                    "preflight",
                    None,
                )
                report = {
                    "command": "preflight",
                    "program": _program_report(program),
                    "runtime_budget": (
                        executor_preflight.model_dump(mode="json")
                        if executor_preflight is not None
                        else None
                    ),
                    "feature_axis": (
                        feature_preflight.model_dump(mode="json")
                        if feature_preflight is not None
                        else None
                    ),
                    "actor_binding_hash": binding.binding_hash,
                    "sealed_preflight_receipt_hash": (
                        getattr(receipt, "receipt_hash", None) if receipt is not None else None
                    ),
                    "stage_telemetry": dict(getattr(executors[0], "stage_telemetry", {})),
                    "preflight_wall_seconds": perf_counter() - command_started,
                    **_call_counts(0),
                }
            elif command == "inspect":
                program, evidence, binding, standing = workflow.inspect(document, **actor)
                report = {
                    **_inspect_report(
                        program, evidence, binding, scientific_disposition=scientific_disposition
                    ),
                    "standing": standing,
                }
            else:
                lifecycle = {
                    "run": workflow.run_sealed,
                    "resume": workflow.resume,
                    "alpha-run": workflow.run_sealed,
                    "alpha-resume": workflow.resume,
                    "score-filter": workflow.run_sealed,
                    "portfolio-run": workflow.run_sealed,
                    "portfolio-resume": workflow.resume,
                    "replay": workflow.replay,
                }
                try:
                    operation = lifecycle[command]
                except KeyError as error:
                    raise AuthoringError("research_authoring.lifecycle_command_invalid") from error
                if command == "alpha-run":
                    # Seal through the already-built executor so the numerical
                    # run consumes the same resolved Feature plan instead of
                    # repeating a standalone preflight in a second process.
                    workflow.preflight(document, **actor)
                evidence, binding = operation(document, **actor)
                stage_telemetry = dict(getattr(executors[0], "stage_telemetry", {}))
                report = {
                    **_evidence_report(
                        command,
                        evidence,
                        binding,
                        numerical_call_count=evidence.numerical_call_count,
                        scientific_disposition=scientific_disposition,
                    ),
                    # Zero only where no executor was resolved. `replay` is that
                    # path; every other command here drives real owners and does
                    # not answer for their counters.
                    **_call_counts(0 if command == "replay" else None),
                    "stage_telemetry": stage_telemetry,
                }
    except AuthoringError as error:
        print(json.dumps({"error": str(error)}, indent=2, sort_keys=True))
        return _exit_code(error)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))
