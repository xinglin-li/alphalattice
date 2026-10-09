"""Bind a prepared local training source to the installed Alpha lifecycle workflow."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from alphalattice.capabilities.alpha_modeling.adapters.lightgbm_dynamic_panel import (
    DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID,
)
from alphalattice.control.product_host.composition.research_authoring import (
    build_research_program_workflow,
)
from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceExperimentInput,
    ResearchWorkspaceModelTrainingInput,
    read_research_workspace_manifest,
    resolve_workspace_model_lifecycle,
)
from alphalattice.control.product_host.research_authoring.authority import (
    WorkspaceResearchAuthorityResolver,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    PreparedInputAuthority,
    normalize_input_document,
    read_factor_bundle,
)
from alphalattice.control.product_host.research_authoring.model_training import (
    model_lifecycle_disclosure,
)
from alphalattice.control.product_host.storage.inventory import (
    StorageInventoryError,
    require_storage_capacity,
)
from alphalattice.control.research_program.authoring.dispatcher import SealedSubmission
from alphalattice.control.research_program.authoring.workflow import ResearchProgramWorkflow
from alphalattice.investment.alpha_research.experiments.lifecycle_authoring import (
    METHOD,
    AlphaLifecycleExperiment,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    DEFAULT_MODEL_LIFECYCLE,
    ModelLifecycle,
    model_lifecycle_of,
)
from alphalattice.protocols.actor_execution.contracts import ActorKind
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    ResearchExperimentEnvelope,
)


def lifecycle_controls(
    workspace: Path,
    binding: ResearchWorkspaceExperimentInput,
    component_id: str | None,
    model_lifecycle: ModelLifecycle | None = None,
) -> dict[str, Any]:
    """Require an explicit prepared component source before building its exact Alpha draft.

    A component prepared under both lifecycles offers the one the request names, else the
    light default; one prepared under a single lifecycle offers it unless another is named.

    Args:
        workspace: Caller-owned admitted workspace.
        binding: Exact admitted research input.
        component_id: Optional explicit component selection.
        model_lifecycle: Optional lifecycle selection, LIGHT or FULL.

    Returns:
        Source-selection request or exact lifecycle document/template and PLAN request; no numerical
        work or strategy/current activation.
    """
    manifest = read_research_workspace_manifest(workspace)
    choices = tuple(
        v
        for v in manifest.model_training_inputs or ()
        if v.input_binding_hash == binding.binding_hash
        and (component_id is None or v.component_id == component_id)
    )
    if component_id is not None and (model_lifecycle is not None or len(choices) > 1):
        wanted = model_lifecycle or DEFAULT_MODEL_LIFECYCLE
        choices = tuple(v for v in choices if _lifecycle_of(workspace, v) == wanted)
        if not choices:
            return {
                "status": "MODEL_TRAINING_SOURCE_SELECTION_REQUIRED",
                "sources": [],
                "claim": f"No source of this component is prepared under the {wanted} lifecycle.",
                "next_requests": {
                    "training": {
                        "operation": "MODEL_TRAINING_INPUT_PLAN",
                        "component_id": component_id,
                        "research_input_id": binding.input_id,
                        "input_binding_hash": binding.binding_hash,
                        "model_lifecycle": wanted,
                    }
                },
            }
    if component_id is None or len(choices) != 1:
        return {
            "status": "MODEL_TRAINING_SOURCE_SELECTION_REQUIRED",
            "sources": [v.model_dump(mode="json") for v in choices],
            "claim": "Select a prepared component source; no implicit latest source or parent.",
        }
    selected = choices[0]
    _store, admission = resolve_workspace_model_lifecycle(
        workspace, component_id=component_id, training_authority_hash=selected.authority_hash
    )
    bundle = read_factor_bundle(workspace, binding.binding_hash)
    days = tuple(
        day
        for day in bundle.sessions
        if admission.formation_start <= day <= admission.formation_end
    )
    document = {
        "experiment": {
            "kind": "alpha.model-development",
            "schema_id": "research-experiment-envelope",
            "data_snapshot_handle": selected.source_handle,
            "universe_handle": "us-current-index-research",
            "sessions": {
                "start": str(days[0]),
                "end": str(days[-1]),
                "as_of": {"session": str(bundle.sessions[-1]), "phase": "OFFICIAL_CLOSE"},
            },
            "budget": {
                "maximum_candidates": admission.component.feature_count,
                "maximum_numerical_calls": admission.maximum_fit_attempts * 2
                + len(days) * len(admission.lifecycle.seeds) * admission.lifecycle.vintage_count,
            },
            "determinism": {
                "seed": admission.lifecycle.seeds[0],
                "thread_limit": 1,
                "network_disabled": True,
            },
            "output_workspace": "managed",
            "baseline_workspace": "managed",
            "publication_intent": "DEVELOPMENT_EVIDENCE_ONLY",
        },
        "alpha": {
            "methodology_id": METHOD,
            "component_recipe_id": component_id,
            "lifecycle": admission.lifecycle.model_dump(mode="json", exclude={"content_hash"}),
        },
    }
    return {
        "status": "DRAFT_READY",
        "document": document,
        "template": document,
        "yaml": yaml.safe_dump(document, sort_keys=False),
        "controls": [],
        "research_input_id": binding.input_id,
        "input_binding_hash": binding.binding_hash,
        "model_training_source": selected.model_dump(mode="json"),
        "model_lifecycle": model_lifecycle_disclosure(
            _lifecycle_of(workspace, selected) or "FULL", admission.lifecycle
        ),
        "plan_request": {
            "operation": "EXPERIMENT_PLAN",
            "research_input_id": binding.input_id,
            "input_binding_hash": binding.binding_hash,
        },
        "numerical_call_count": 0,
        "limits": [
            "LOCAL_FROZEN_RECIPE_RECONSTRUCTION_NOT_ORIGINAL_RESEARCH_RESULTS",
            "NO_STRATEGY_OR_CURRENT_ACTIVATION",
        ],
    }


def _lifecycle_of(
    workspace: Path, source: ResearchWorkspaceModelTrainingInput
) -> ModelLifecycle | None:
    """Which installed lifecycle a prepared training source's admission binds."""
    _store, admission = resolve_workspace_model_lifecycle(
        workspace, component_id=source.component_id, training_authority_hash=source.authority_hash
    )
    return model_lifecycle_of(
        INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(source.component_id),
        admission.lifecycle.content_hash,
    )


def lifecycle_workflow(
    *,
    workspace: Path,
    binding: ResearchWorkspaceExperimentInput,
    document: dict[str, Any],
    cancelled: Callable[[], bool] = lambda: False,
) -> tuple[
    ResearchProgramWorkflow,
    SealedSubmission,
    dict[str, Any],
    dict[str, Any],
    ResearchWorkspaceModelTrainingInput,
]:
    """Require exact prepared component authority and build its bounded Alpha lifecycle workflow.

    Args:
        workspace: Caller-owned admitted workspace.
        binding: Exact admitted research input.
        document: Explicit authored lifecycle declaration.
        cancelled: Explicit cancellation callback.

    Returns:
        Workflow, sealed preview, lifecycle work bounds, normalized document and exact selected
        model input.

    Raises:
        AuthoringError: Source selection, lifecycle interval or required storage capacity is
            inadmissible.
    """
    manifest = read_research_workspace_manifest(workspace)
    handle = document.get("experiment", {}).get("data_snapshot_handle")
    matches = tuple(
        v
        for v in manifest.model_training_inputs or ()
        if v.source_handle == handle and v.input_binding_hash == binding.binding_hash
    )
    if len(matches) != 1:
        raise AuthoringError("alpha_research.lifecycle_source_selection_required")
    selected: ResearchWorkspaceModelTrainingInput = matches[0]
    store, admission = resolve_workspace_model_lifecycle(
        workspace,
        component_id=selected.component_id,
        training_authority_hash=selected.authority_hash,
    )
    bundle = read_factor_bundle(workspace, binding.binding_hash)
    normalized = normalize_input_document(
        document,
        binding,
        bundle,
        section_name="alpha",
        source_handle=selected.source_handle,
        baseline_path=store.root.relative_to(workspace).as_posix(),
    )
    envelope = ResearchExperimentEnvelope.create(**normalized["experiment"])
    if (
        envelope.sessions.start < admission.formation_start
        or envelope.sessions.end > admission.formation_end
    ):
        raise AuthoringError("alpha_research.lifecycle_interval_outside_admitted_source")
    authority = WorkspaceResearchAuthorityResolver(workspace=workspace).resolve(envelope)

    def capacity(amount: int) -> None:
        try:
            require_storage_capacity(workspace, additional_bytes=amount)
        except StorageInventoryError as error:
            raise AuthoringError(error.failure_code) from error

    method = AlphaLifecycleExperiment(
        store,
        admission,
        cancelled=cancelled,
        sharing_workspace=workspace,
        packed_capacity=capacity,
    )
    workflow = build_research_program_workflow(
        workspace=store.root,
        workspace_root=workspace,
        executors=(method,),
        compilers=(method,),
        authority=PreparedInputAuthority(envelope, authority),
        verifier_kinds=(method.kind,),
    )
    sealed = workflow.prepare(normalized, actor_kind=ActorKind.HUMAN, actor_id="preview")
    assert method.preflight is not None
    preview = {
        "statistical_start": str(authority.sessions[0]),
        "statistical_end": str(authority.sessions[-1]),
        "statistical_session_count": len(authority.sessions),
        "interval_semantics": "COMPLETE_ADMITTED_MODEL_LIFECYCLE_FORMATIONS",
        "listing_count": len(authority.ordered_listing_ids),
        "methodology_id": METHOD,
        "model_adapter_id": DYNAMIC_PANEL_LIGHTGBM_ADAPTER_ID,
        "component_id": selected.component_id,
        "ordered_feature_ids": list(admission.component.ordered_feature_ids),
        "lifecycle": method.preflight.lifecycle.model_dump(mode="json"),
        "required_vintages": list(method.preflight.required_vintages),
        "fit_upper_bound": method.preflight.fit_upper_bound,
        "prediction_call_upper_bound": method.preflight.prediction_call_upper_bound,
        "expected_numerical_calls": method.preflight.fit_upper_bound * 2
        + method.preflight.prediction_call_upper_bound,
    }
    return workflow, sealed, preview, normalized, selected
