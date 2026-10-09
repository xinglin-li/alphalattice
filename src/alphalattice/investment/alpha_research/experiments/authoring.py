"""Compile one authored Alpha experiment into a typed development Program.

The Alpha Desk already owned a target catalog, a model catalog, a capability
mandate and a fold-at-a-time executor. What it did not own was a way for an
*authored document* to select among them, so the only caller of the development
target recipe was a test that constructed one by hand.

This compiler is that selection, and nothing more. Every name in the document has
to resolve against something the Host installed -- a target recipe id, a model
capability handle, model parameters inside that capability's declared search
domain, and an ordered feature axis the Host has already checked against both the
published Panel and the Factor development evidence. A name that resolves to
nothing fails here, before any authority is bound and long before any fit.

The compiler produces identity, never scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from alphalattice.capabilities.alpha_modeling.catalog import AlphaModelCatalog
from alphalattice.investment.alpha_research.experiments.mandate import (
    AlphaModelCapabilityAuthority,
    AlphaModelRecipeProposal,
    AlphaResearchModelRecipe,
)
from alphalattice.investment.alpha_research.experiments.panel_methodology_authoring import (
    PanelResearchMethodologyCompiler,
    methodology_section,
)
from alphalattice.investment.alpha_research.targets.authority import (
    AlphaTargetMethod,
    AlphaTargetMethodCatalog,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.research_authoring.contracts import (
    AuthoringError,
    DeskProgramCompilation,
    DeskSection,
    ResearchExperimentEnvelope,
    ResolvedResearchAuthority,
    refuse_unknown_section_keys,
)

ALPHA_EXPERIMENT_KIND = "alpha.model-development"


class AlphaDevelopmentSection(DeskSection):
    """An Alpha model study's section, one that names no `methodology_id`.

    The compiler reads the first four keys; the Host's execution reads the rest, which its
    drafts write.
    """

    target_recipe_id: str
    """The installed target method the model learns."""
    ordered_feature_ids: list[str]
    """The Factors the model reads, in the order its arrays hold them."""
    model_capability_handle: str
    """The installed model capability."""
    model_parameters: dict[str, Any]
    """The capability's parameters, admitted by its search domain."""
    factor_evidence_handle: str
    """The Factor evidence that screened the study's Factors; the Host's draft writes it."""
    foundation_admission_hash: str | None = None
    """The sealed Foundation the study is built on, when a Foundation's draft wrote it."""
    execution_outcome_recipe_id: str | None = None
    """The execution outcome recipe a direct run declares, checked against the target's."""
    causal_outcome_snapshot_handle: str | None = None
    """The causal outcome snapshot a direct run reads, when it names one."""


MAXIMUM_DEVELOPMENT_FEATURE_AXIS = 55
"""The same bound the frozen research foundation applies to a feature axis.

An axis over the bound is **rejected**, never sliced. Truncation would produce a
different experiment that still looks like the one somebody authored: the Program
would name 55 factors, the document would name more, and both would hash to
something self-consistent.
"""


_MODEL_DOMAIN_GUARD = {
    "guard": "MODEL_DOMAIN",
    "refusal": "alpha_research.authoring_model_recipe_not_admissible",
}
_BOUNDED_HELP = "Must lie in the installed model domain; values are never silently clipped."

# The Dynamic Panel LightGBM parameters an author states, in recipe order: the
# control label, the unit, the constraint prefix the domain bounds it with, and
# the step (``1`` for a count, ``"any"`` for a real). ``training_policy`` and
# ``fixed_iterations`` follow separately: one is a choice, the other is a count
# only the fixed-iteration policy reads.
_DYNAMIC_PANEL_LIGHTGBM_NUMBERS: tuple[tuple[str, str, str | None, str, int | str], ...] = (
    ("seed", "Random seed", None, "seed", 1),
    ("max_depth", "Maximum tree depth", "levels", "max_depth", 1),
    ("num_leaves", "Leaves per tree", "leaves", "num_leaves", 1),
    ("min_child_samples", "Minimum rows per leaf", "rows", "min_child_samples", 1),
    ("learning_rate", "Learning rate", "ratio", "learning_rate", "any"),
    ("lambda_l1", "L1 regularization", None, "lambda", "any"),
    ("lambda_l2", "L2 regularization", None, "lambda", "any"),
    ("min_gain_to_split", "Minimum gain to split", None, "min_gain_to_split", "any"),
    ("feature_fraction", "Feature fraction per tree", "ratio", "fraction", "any"),
    ("bagging_fraction", "Row fraction per bag", "ratio", "fraction", "any"),
    ("bagging_freq", "Bagging frequency", "iterations", "bagging_freq", 1),
)


def _model_parameter_controls(
    adapter_id: str, bounds: Mapping[str, Any], selected: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Project one installed capability's parameter domain as authoring controls.

    ``selected`` is the condition under which these controls apply: the
    document names this capability's handle. A control the page shows only for
    some values of another parameter carries that condition too; every
    condition in ``when`` must hold for the control to be visible.
    """

    parameters = ["alpha", "model_parameters"]
    if adapter_id == "regularized_linear":
        controls: list[dict[str, Any]] = [
            {
                "path": [*parameters, "family"],
                "label": "Linear model family",
                "type": "select",
                "options": bounds["allowed_families"],
                "unit": None,
                "when": [dict(selected)],
                **_MODEL_DOMAIN_GUARD,
                "help": "One explicit model configuration; no automatic search.",
            }
        ]
        for name, label, families, prefix in (
            ("alpha", "Ridge regularization", ["ridge"], "ridge_alpha"),
            (
                "alpha_max_multiplier",
                "Sparse regularization multiplier",
                ["lasso", "elastic_net"],
                "sparse_alpha_max_multiplier",
            ),
            ("l1_ratio", "Elastic Net L1 ratio", ["elastic_net"], "elastic_net_l1_ratio"),
        ):
            controls.append(
                {
                    "path": [*parameters, name],
                    "label": label,
                    "type": "number",
                    "when": [dict(selected), {"path": [*parameters, "family"], "values": families}],
                    "unit": "ratio",
                    "step": "any",
                    "min": bounds[prefix + "_min"],
                    "max": bounds[prefix + "_max"],
                    **_MODEL_DOMAIN_GUARD,
                    "help": _BOUNDED_HELP,
                }
            )
        return controls
    if adapter_id == "dynamic_panel_lightgbm":
        controls = [
            {
                "path": [*parameters, name],
                "label": label,
                "type": "number",
                "when": [dict(selected)],
                "unit": unit,
                "step": step,
                "min": bounds[prefix + "_min"],
                "max": bounds[prefix + "_max"],
                **_MODEL_DOMAIN_GUARD,
                "help": _BOUNDED_HELP,
            }
            for name, label, unit, prefix, step in _DYNAMIC_PANEL_LIGHTGBM_NUMBERS
        ]
        controls.append(
            {
                "path": [*parameters, "training_policy"],
                "label": "Training policy",
                "type": "select",
                "options": list(bounds["allowed_training_policies"]),
                "unit": None,
                "when": [dict(selected)],
                **_MODEL_DOMAIN_GUARD,
                "help": (
                    "FIXED_ITERATION fits the stated iteration count on the whole training "
                    "surface; the early-stopping policies need a tuning partition this "
                    "development split does not state and are refused at PLAN."
                ),
            }
        )
        controls.append(
            {
                "path": [*parameters, "fixed_iterations"],
                "label": "Boosting iterations",
                "type": "number",
                "when": [
                    dict(selected),
                    {"path": [*parameters, "training_policy"], "values": ["FIXED_ITERATION"]},
                ],
                "unit": "iterations",
                "step": 1,
                "min": bounds["fixed_iterations_min"],
                "max": bounds["fixed_iterations_max"],
                **_MODEL_DOMAIN_GUARD,
                "help": _BOUNDED_HELP,
            }
        )
        return controls
    # An installed capability this projection has no controls for still admits
    # YAML declarations through the mandate; the page keeps YAML editing for it.
    return []


class AlphaExperimentCompiler:
    """Desk-owned compiler; selects installed methodology and seals its identity.

    All four authorities are constructor state supplied by the Host, for the same
    reason the Factor inventory is: which target recipes are installed, which
    model capabilities are mandated, which factors the Panel published and which
    the Factor development evidence answered for are facts about a workspace and
    a build, not literals a Desk module may carry.
    """

    kind = ALPHA_EXPERIMENT_KIND

    def __init__(
        self,
        *,
        target_recipes: AlphaTargetMethodCatalog,
        model_mandate: AlphaModelCapabilityAuthority,
        model_catalog: AlphaModelCatalog,
        panel_factor_ids: tuple[str, ...],
        factor_evidence_factor_ids: tuple[str, ...],
        factor_evidence_checkpoint_hash: str,
        research_foundation_hash: str | None = None,
        methodology_compiler: PanelResearchMethodologyCompiler | None = None,
    ) -> None:
        self._target_recipes = target_recipes
        self._model_mandate = model_mandate
        self._model_catalog = model_catalog
        self._panel_factor_ids = frozenset(panel_factor_ids)
        self._evidence_factor_ids = frozenset(factor_evidence_factor_ids)
        self._evidence_checkpoint_hash = factor_evidence_checkpoint_hash
        self._research_foundation_hash = research_foundation_hash
        self._frozen_feature_axis = factor_evidence_factor_ids if research_foundation_hash else None
        self._methodology_compiler = methodology_compiler

    def authoring_options(self) -> dict[str, Any]:
        """Expose the installed declaration space without selecting a recipe."""
        domains = self._model_mandate.ordered_search_domains
        controls: list[dict[str, Any]] = [
            {
                "path": ["alpha", "target_recipe_id"],
                "label": "Alpha target",
                "type": "select",
                "options": list(self._target_recipes.method_ids),
                "unit": None,
                "help": "Select an installed transformation of the bound causal outcomes.",
                "guard": "INSTALLED_TARGET",
                "refusal": "alpha_research.authoring_target_recipe_not_installed",
            },
            {
                "path": ["alpha", "model_capability_handle"],
                "label": "Model capability",
                "type": "select",
                "options": [
                    {"value": self._model_mandate.capability_handle(i), "label": d.adapter_id}
                    for i, d in enumerate(domains)
                ],
                "unit": None,
                "help": "The installed capability admits the parameter declaration.",
                "guard": "INSTALLED_MODEL",
                "refusal": "alpha_research.authoring_model_recipe_not_admissible",
            },
            {
                "path": ["experiment", "budget", "maximum_numerical_calls"],
                "label": "Numerical call budget",
                "type": "number",
                "unit": "calls",
                "min": 1,
                "max": 1_000_000,
                "step": 1,
                "help": "Fit, prediction and metric work; preview shows the admitted upper bound.",
                "guard": "PRE_FIT_BUDGET",
                "refusal": "alpha_research.development_work_budget_exceeded",
            },
        ]
        # This is a projection of the installed parameter domain, not another
        # admission rule. The model adapter still validates every declaration.
        # Each capability's controls are shown while its handle is the one the
        # document names, so a page reading this inventory shows every installed
        # model, not only the linear one.
        for index, domain in enumerate(domains):
            handle = self._model_mandate.capability_handle(index)
            selected = {"path": ["alpha", "model_capability_handle"], "values": [handle]}
            controls.extend(
                _model_parameter_controls(domain.adapter_id, domain.constraints, selected)
            )
        return {
            "controls": controls,
            "target_recipe_ids": list(self._target_recipes.method_ids),
            "model_capabilities": [
                {
                    "handle": self._model_mandate.capability_handle(index),
                    "domain": domain.model_dump(mode="json"),
                }
                for index, domain in enumerate(self._model_mandate.ordered_search_domains)
            ],
        }

    def compile_desk_program(
        self,
        *,
        envelope: ResearchExperimentEnvelope,
        document: Mapping[str, Any],
        authority: ResolvedResearchAuthority,
    ) -> DeskProgramCompilation:
        section = document.get("alpha")
        if not isinstance(section, dict):
            raise AuthoringError("alpha_research.authoring_section_missing")
        selected_methodology = methodology_section(document)
        if authority.panel_snapshot_hash is None:
            raise AuthoringError("alpha_research.panel_authority_required")
        if selected_methodology is not None:
            if self._methodology_compiler is None:
                raise AuthoringError("alpha_research.methodology_not_installed")
            return self._methodology_compiler.compile(
                envelope=envelope,
                section=selected_methodology,
                authority=authority,
            )

        refuse_unknown_section_keys(section, AlphaDevelopmentSection, place="alpha")
        method = self._resolve_target_method(section)
        ordered_feature_ids = self._resolve_feature_axis(section)
        # Admitted against exactly the lanes the resolved method admits, as
        # execution admits it. A compiler that admitted differently would seal a
        # different ``research_recipe_hash`` than the one that runs, and the run
        # would fail its own identity check for a reason no message explains.
        model_recipe = self._admit_model_recipe(section, method=method)

        if len(ordered_feature_ids) > envelope.budget.maximum_candidates:
            # The authored budget is a second, tighter bound than the axis cap:
            # a document may declare it will look at fewer factors than the Host
            # would allow, and that declaration is part of what it authored.
            raise AuthoringError("alpha_research.authoring_feature_budget_exceeded")

        # The installed catalogs as far as this Program reaches them: the target
        # method it names and the one model its mandate admits for it. The other
        # recipes and models installed beside them decide none of its numbers, so
        # installing one moves no Program (as the panel methodology's
        # per-method catalog already did).
        catalog_hash = str(
            canonical_hash(
                {
                    "target_method_hash": method.target_method_hash,
                    "model_mandate_hash": self._model_mandate.admitting(
                        model_recipe.search_domain_hash
                    ).mandate_hash,
                }
            )
        )
        # The admissible space of the chosen method: the target recipe it names
        # and the domain its model's parameters were admitted from, not the menu
        # of every installed recipe and mandated model.
        parameter_domain_hash = str(
            canonical_hash(
                {
                    "target_recipe_id": method.target_recipe_id,
                    "search_domain_hash": model_recipe.search_domain_hash,
                }
            )
        )
        # The method actually selected, complete enough that no part of it can be
        # chosen later without moving this hash: the target recipe including its
        # named standardization, the admitted model recipe including its
        # parameters and the domain that admitted them, and the ordered feature
        # axis the run will read.
        method_binding_hash = str(
            canonical_hash(
                {
                    "target_recipe_id": method.target_recipe_id,
                    "target_method_hash": method.target_method_hash,
                    "standardization_id": method.standardization_id,
                    "research_recipe_hash": model_recipe.research_recipe_hash,
                    "search_domain_hash": model_recipe.search_domain_hash,
                    "ordered_feature_ids": list(ordered_feature_ids),
                    "factor_evidence_checkpoint_hash": self._evidence_checkpoint_hash,
                    **(
                        {"research_foundation_hash": self._research_foundation_hash}
                        if self._research_foundation_hash is not None
                        else {}
                    ),
                    "parameter_domain_hash": parameter_domain_hash,
                }
            )
        )
        desk_program_hash = str(
            canonical_hash(
                {
                    "kind": self.kind,
                    "envelope_hash": envelope.envelope_hash,
                    "method_binding_hash": method_binding_hash,
                    "catalog_hash": catalog_hash,
                    "authority_hash": authority.authority_hash,
                }
            )
        )
        return DeskProgramCompilation(
            desk_program_hash=desk_program_hash,
            catalog_hash=catalog_hash,
            method_binding_hash=method_binding_hash,
            parameter_domain_hash=parameter_domain_hash,
        )

    def _resolve_target_method(self, section: Mapping[str, Any]) -> AlphaTargetMethod:
        recipe_id = section.get("target_recipe_id")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise AuthoringError("alpha_research.authoring_target_recipe_id_invalid")
        try:
            return self._target_recipes.resolve(recipe_id)
        except ValueError as error:
            raise AuthoringError("alpha_research.authoring_target_recipe_not_installed") from error

    def _resolve_feature_axis(self, section: Mapping[str, Any]) -> tuple[str, ...]:
        """Admit the declared axis, exactly as declared or not at all.

        Order is admitted rather than repaired because every array surface
        downstream is positional: a permuted axis and a sorted axis are two
        different matrices, and silently sorting one into the other is how a
        model gets trained against columns it was not given.
        """

        declared = section.get("ordered_feature_ids")
        if not isinstance(declared, Sequence) or isinstance(declared, str | bytes) or not declared:
            raise AuthoringError("alpha_research.authoring_feature_axis_invalid")
        ordered = tuple(str(value) for value in declared)
        if self._frozen_feature_axis is not None and ordered != self._frozen_feature_axis:
            raise AuthoringError("research_foundation.feature_axis_frozen")
        if ordered != tuple(dict.fromkeys(ordered)):
            raise AuthoringError("alpha_research.authoring_feature_axis_duplicated")
        if len(ordered) > MAXIMUM_DEVELOPMENT_FEATURE_AXIS:
            raise AuthoringError("alpha_research.authoring_feature_axis_exceeds_bound")
        if not set(ordered).issubset(self._panel_factor_ids):
            raise AuthoringError("alpha_research.authoring_feature_axis_not_in_panel")
        if not set(ordered).issubset(self._evidence_factor_ids):
            # The Factor development evidence is what authorized these factors
            # for research use. A factor the Panel published but the evidence
            # never answered for has no development standing here.
            raise AuthoringError("alpha_research.authoring_feature_axis_not_in_factor_evidence")
        return ordered

    def _admit_model_recipe(
        self, section: Mapping[str, Any], *, method: AlphaTargetMethod
    ) -> AlphaResearchModelRecipe:
        handle = section.get("model_capability_handle")
        if not isinstance(handle, str) or not handle:
            raise AuthoringError("alpha_research.authoring_model_capability_handle_invalid")
        parameters = section.get("model_parameters")
        if not isinstance(parameters, dict):
            raise AuthoringError("alpha_research.authoring_model_parameters_invalid")
        try:
            # The mandate admits, not this compiler: it resolves the handle,
            # routes the parameters through the capability's declared search
            # domain and refuses anything outside it. Re-checking here would be a
            # second opinion about admissibility, which is exactly the drift the
            # capability mandate exists to prevent.
            lanes = method.admitted_model_lanes
            return self._model_mandate.admit_proposal(
                proposal=AlphaModelRecipeProposal(
                    capability_handle=handle,
                    parameters=dict(parameters),
                    target_lane=None if lanes is None else lanes[0],
                ),
                catalog=self._model_catalog,
                admitted_target_lanes=lanes,
            )
        except ValueError as error:
            raise AuthoringError("alpha_research.authoring_model_recipe_not_admissible") from error


__all__ = [
    "ALPHA_EXPERIMENT_KIND",
    "MAXIMUM_DEVELOPMENT_FEATURE_AXIS",
    "AlphaDevelopmentSection",
    "AlphaExperimentCompiler",
]
