"""One component's model lifecycle admission from its frozen training source (O3, V325).

Moved from the Host's model training preparation: which training vintages a component's
lifecycle needs, which of them its source supports, and the admission that binds them are
Alpha's; the Host reads the workspace's inputs and records the admissions it returns. It moved
with MODEL_KERNEL's rotation for the dead code V365 and V366 took out, so the domain moved once.
"""

from __future__ import annotations

from collections.abc import Callable

from alphalattice.investment.alpha_research.inputs.frozen_price_volume import (
    FrozenPriceVolumeInputs,
)
from alphalattice.investment.alpha_research.inputs.panel_feature_views import (
    PanelFeatureSourceArrays,
)
from alphalattice.investment.alpha_research.publication.artifacts import AlphaCurrentArtifactStore
from alphalattice.investment.alpha_research.scores.heterogeneous_product import (
    HeterogeneousAlphaComponentRecipe,
)
from alphalattice.investment.alpha_research.scores.model_renewal import (
    AlphaModelLifecycleAdmission,
    prepare_alpha_refit,
    publish_component_training_observations,
)
from alphalattice.investment.alpha_research.scores.product_lifecycle import (
    AlphaModelLifecycleRecipe,
    resolve_alpha_refit_plan,
)


def prepare_component_lifecycle(
    store: AlphaCurrentArtifactStore,
    *,
    component: HeterogeneousAlphaComponentRecipe,
    source: FrozenPriceVolumeInputs,
    training: PanelFeatureSourceArrays,
    environment_hash: str,
    cancelled: Callable[[], bool],
    progress: Callable[[str], None],
) -> tuple[AlphaModelLifecycleAdmission, tuple[tuple[str, str], ...]]:
    """Publish one component's training observations, refit plans and lifecycle admission.

    Every training vintage the component's lifecycle needs over the source's formations is
    preflighted, with no fit; a vintage the source cannot support is set aside with its code,
    and the admission binds the contiguous formations whose vintages are all prepared.

    Args:
        store: The workspace's current Alpha artifacts.
        component: The installed component recipe.
        source: The frozen price and volume inputs.
        training: The component's training source arrays.
        environment_hash: The prediction owner's environment the admission records.
        cancelled: Asked at each vintage's safe checkpoint.
        progress: Told each vintage it preflights.

    Returns:
        The published admission, and each vintage set aside with its refusal code.

    Raises:
        ValueError: Cancelled at a checkpoint, no complete model history, a gap inside the
            support, or a refit refusal other than a vintage's missing support.
    """
    observations = publish_component_training_observations(
        store,
        component=component,
        source=source,
        training=training,
    )
    rule = AlphaModelLifecycleRecipe.from_component(component)
    prepared = []
    rejected: list[tuple[str, str]] = []
    vintages = sorted({v for day in source.formation_sessions for v in rule.vintages(day)})
    for vintage in vintages:
        if cancelled():
            raise ValueError("model_training.cancelled_at_safe_checkpoint")
        progress(f"{component.component_id}: preflight training vintage {vintage}; no fit")
        try:
            plan = resolve_alpha_refit_plan(
                lifecycle=rule,
                vintage=vintage,
                sessions=source.formation_sessions,
                component_recipe_hash=component.recipe_hash,
                source_binding_hash=observations.content_hash,
                ordered_listing_ids=source.ordered_listing_ids,
                ordered_feature_ids=component.ordered_feature_ids,
            )
            prepared.append(
                prepare_alpha_refit(
                    store, plan=plan, observations=observations, component=component
                )
            )
        except ValueError as error:
            code = str(error)
            if code not in {
                "alpha_research.refit_history_insufficient",
                "alpha_research.refit_period_unavailable",
                "alpha_research.refit_training_support_insufficient",
                "alpha_research.heterogeneous_live_trend_candidate_support_insufficient",
            }:
                raise
            rejected.append((vintage, code))
    admitted_vintages = {value.plan.vintage for value in prepared}
    supported = tuple(
        day for day in source.formation_sessions if set(rule.vintages(day)) <= admitted_vintages
    )
    if not supported:
        raise ValueError("model_training.complete_model_history_unavailable")
    if supported != tuple(day for day in source.formation_sessions if day >= supported[0]):
        raise ValueError("model_training.interior_model_support_gap")
    required = {v for day in supported for v in rule.vintages(day)}
    used = tuple(value for value in prepared if value.plan.vintage in required)
    admission = AlphaModelLifecycleAdmission.create(
        component=component,
        lifecycle=rule,
        observations_hash=observations.content_hash,
        prepared=used,
        initial_children=(),
        formation_start=supported[0],
        formation_end=supported[-1],
        maximum_fit_attempts=len(required) * len(rule.seeds),
        environment_hash=environment_hash,
        fit_vintages=tuple(sorted(required)),
    )
    store._publish("lifecycle-admissions", admission, "content_hash")
    return admission, tuple(rejected)


__all__ = ["prepare_component_lifecycle"]
