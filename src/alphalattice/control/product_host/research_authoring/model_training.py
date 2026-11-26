"""Prepare a local frozen-component training source; never fit or install a strategy."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from alphalattice.control.product_host.composition.research_workspace import (
    ResearchWorkspaceModelTrainingInput,
)
from alphalattice.control.product_host.composition.strategy_score_inputs import (
    read_workspace_component_inputs,
    workspace_score_source_identity,
)
from alphalattice.control.product_host.research_authoring.factor_inputs import (
    factor_input_paths,
    read_factor_bundle,
)
from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY,
)
from alphalattice.investment.alpha_research.scores.heterogeneous_replay import (
    AlphaRuntimeHeterogeneousPredictionOwner,
)
from alphalattice.investment.alpha_research.scores.lifecycle_preparation import (
    prepare_component_lifecycle,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
    ModelLifecycle,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.kernel.shared_kernel.sealing import seal_model
from alphalattice.kernel.shared_kernel.spans import span

# This is the repaired fixed sparse-source axis, not today's number of formulas.
# Existing owners supply the IDs; the pin prevents a later catalogue addition
# from silently changing the frozen component's common training-row denominator.
FROZEN_TRAINING_FACTOR_AXIS_HASH = (
    "90e032fec9de4e23930569f292a8bb6fa076357de3ae4ff22b1b1213d5a302ce"
)


class ComponentTrainingPreparationReceipt(BaseModel):  # type: ignore[misc]
    """Seal exact component input authorities, source/factor axes and rejected vintages."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    input_binding_hash: str
    source_identity_hash: str
    training_factor_ids: tuple[str, ...]
    bindings: tuple[ResearchWorkspaceModelTrainingInput, ...]
    rejected_vintages: tuple[tuple[str, str, str], ...]
    receipt_hash: str

    @classmethod
    def create(cls, **values: Any) -> Self:
        """Seal an explicit component input preparation receipt.

        Args:
            values: Explicit model fields excluding the generated self identity.

        Returns:
            Validated model with canonical receipt_hash; construction grants no execution or
            publication authority.

        Raises:
            pydantic.ValidationError: Fields or declared consistency violate the concrete model.
        """
        return seal_model(cls, values, field="receipt_hash")

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def verify(self) -> Self:
        """Require exact receipt, frozen training axis and all component source bindings.

        Returns:
            This contract after its declared consistency checks.

        Raises:
            ValueError: Canonical receipt, frozen factor axis or component input/source identities
                differ.
        """
        if (
            self.receipt_hash
            != canonical_hash(self.model_dump(mode="json", exclude={"receipt_hash"}))
            or canonical_hash(self.training_factor_ids) != FROZEN_TRAINING_FACTOR_AXIS_HASH
            or any(
                v.input_binding_hash != self.input_binding_hash
                or v.source_identity_hash != self.source_identity_hash
                for v in self.bindings
            )
        ):
            raise ValueError("model_training.preparation_receipt_invalid")
        return self


def model_lifecycle_disclosure(
    configuration: ModelLifecycle, rule: AlphaModelLifecycleRecipe
) -> dict[str, object]:
    """What a training plan or study's controls say of the lifecycle they prepare.

    The light default is said to be light, with how to ask for the component's full one.
    """
    return {
        "configuration": configuration,
        "seeds": len(rule.seeds),
        "refit_months": rule.month_interval,
        "vintages": rule.vintage_count,
        "live_models": len(rule.seeds) * rule.vintage_count,
        "lifecycle_hash": rule.content_hash,
        "claim": (
            "LIGHT_DEFAULT_FEWER_MODELS_THAN_THE_COMPONENT_FULL_LIFECYCLE; plan model_lifecycle "
            "FULL by name when the person asks for the full one"
            if configuration == "LIGHT"
            else "FULL_COMPONENT_LIFECYCLE_CHOSEN_BY_NAME"
        ),
    }


def frozen_training_factor_ids() -> tuple[str, ...]:
    """Resolve the installed ordered training factor set and require its frozen axis binding.

    Returns:
        Sorted unique installed training factor identities.

    Raises:
        ValueError: Installed factor axis differs from its frozen training binding.
    """
    ids = tuple(
        sorted(
            {
                *desktop_core_feature_bundle().factor_ids,
                *SPARSE_SESSION_AMPLITUDE_REQUIRED_FORMULA_IDS,
            }
        )
    )
    if canonical_hash(ids) != FROZEN_TRAINING_FACTOR_AXIS_HASH:
        raise ValueError("model_training.frozen_source_factor_axis_changed")
    return ids


def prepare_component_training_inputs(
    *,
    workspace: Path,
    input_binding_hash: str,
    component_ids: tuple[str, ...],
    observed_at: datetime,
    cancelled: Callable[[], bool] = lambda: False,
    progress: Callable[[str], None] = lambda _message: None,
    capacity: Callable[[int], None] = lambda _bytes: None,
    lifecycle: ModelLifecycle,
) -> ComponentTrainingPreparationReceipt:
    """Build real inputs and preflight every supported vintage before model work.

    No imported payloads, shortened training windows or implicit current/default
    selection. A leading interval may lack the frozen history requirement;
    a gap after scoring becomes supported refuses the whole preparation.
    """
    if (
        not component_ids
        or len(set(component_ids)) != len(component_ids)
        or set(component_ids) - {"G2_R0_TREND", "G6_R0_FAST_REBOUND"}
    ):
        raise ValueError("model_training.component_selection_invalid")
    components = tuple(
        INSTALLED_HETEROGENEOUS_ALPHA_STRATEGY.component(name) for name in component_ids
    )
    with span("verify", "factor_bundle"):
        bundle = read_factor_bundle(workspace, input_binding_hash)
    root, _ = factor_input_paths(workspace, input_binding_hash)
    with span("hash", "score_source_identity"):
        source_identity = workspace_score_source_identity(root)
    factor_ids = frozen_training_factor_ids()
    features = tuple(
        sorted({name for component in components for name in component.ordered_feature_ids})
    )
    if cancelled():
        raise ValueError("model_training.cancelled_at_safe_checkpoint")
    progress("materialize declared formulas and causal Context from sealed input")
    with span("materialize", "component_inputs"):
        source, training = read_workspace_component_inputs(
            root,
            formation=bundle.sessions[-1],
            observed_at=observed_at,
            expected_source_hash=source_identity,
            ordered_feature_ids=features,
            training_factor_ids=factor_ids,
        )
    if training is None or training.ordered_factor_ids != factor_ids:
        raise ValueError("model_training.training_source_axis_mismatch")
    store = AlphaCurrentArtifactStore(workspace / "artifacts", packed_capacity=capacity)
    if not store.root.is_relative_to(workspace.resolve()):
        raise ValueError("model_training.artifact_root_outside_workspace")
    environment = AlphaRuntimeHeterogeneousPredictionOwner(
        component_ids=component_ids
    ).environment_hash
    result = []
    rejected_vintages: list[tuple[str, str, str]] = []
    for component in components:
        if cancelled():
            raise ValueError("model_training.cancelled_at_safe_checkpoint")
        with span("features", "component_lifecycle"):
            admission, rejected = prepare_component_lifecycle(
                store,
                component=component,
                source=source,
                training=training,
                environment_hash=environment,
                cancelled=cancelled,
                progress=progress,
                lifecycle=lifecycle,
            )
        rejected_vintages.extend(
            (component.component_id, vintage, code) for vintage, code in rejected
        )
        relative = (
            store._path("current/lifecycle-admissions", admission.content_hash, "json")
            .relative_to(workspace.resolve())
            .as_posix()
        )
        result.append(
            ResearchWorkspaceModelTrainingInput(
                component_id=component.component_id,
                input_binding_hash=input_binding_hash,
                source_identity_hash=source_identity,
                authority_relative_path=relative,
                authority_hash=admission.content_hash,
            )
        )
    receipt = ComponentTrainingPreparationReceipt.create(
        input_binding_hash=input_binding_hash,
        source_identity_hash=source_identity,
        training_factor_ids=factor_ids,
        bindings=tuple(result),
        rejected_vintages=tuple(rejected_vintages),
    )
    store._publish("lifecycle-input-preparations", receipt, "receipt_hash")
    return receipt
