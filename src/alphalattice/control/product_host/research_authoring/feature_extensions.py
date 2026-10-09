"""The formula factors a workspace activates into its daily catalog: review, activation (EX).

An agent declares a formula factor in a research plan (`feature plan`), builds and tries it
(`feature run`, `trial run`); a person reads its review packet (`feature review`) and activates
it into this workspace's daily Feature catalog, or deactivates it. Activation records who and
when, the factor's spec and identity, the recipe its declaration chose, the trial it passed and the
packet read; it never edits a sealed record. The workspace's registry,
`runtime/extensions/features.json`, is what its daily catalog reads (`activated_feature_specs`),
so an activated factor is computed by this workspace's daily updates alone.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from alphalattice.control.product_host.composition.feature_trials import FeatureTrial, FeatureTrials
from alphalattice.control.product_host.composition.result_standing import packet_standing
from alphalattice.control.product_host.publication.goals import GoalStore
from alphalattice.control.product_host.research_authoring.factor_inputs import read_factor_bundle
from alphalattice.control.product_host.research_authoring.feature_activations import (
    FeatureActivation,
    FeatureExtensionRegistry,
)
from alphalattice.control.product_host.research_authoring.feature_research import CATEGORY
from alphalattice.foundation.feature_engine.catalog.contracts import desktop_core_feature_bundle
from alphalattice.foundation.feature_engine.catalog.research import (
    ResearchFeaturePlan,
    research_feature_execution_spec,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.core_bundle import (
    numerical_spec_hash,
)
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    FORMULA_IDS,
    FORMULA_RESEARCH_IDS,
    formula_controls,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    factor_methodology_hash,
)
from alphalattice.foundation.feature_engine.producers.factors.specifications import (
    FactorFormulaSpecification,
    build_research_formula_specification,
    golden_example_frame,
)
from alphalattice.foundation.feature_engine.producers.preprocessing.catalog import (
    build_installed_panel_preprocessing_catalog,
)
from alphalattice.interface.local_application.cli_contract import refusal_words
from alphalattice.interface.local_application.failure_codes import public_failure
from alphalattice.kernel.quant.factor_contracts import FactorSpec
from alphalattice.kernel.shared_kernel.environment import recorded_environment
from alphalattice.kernel.shared_kernel.identity import canonical_hash

_LISTING_PAGE = 50
"""Formula factors one page of `FEATURE_EXTENSIONS` lists: the Local Web's table page."""


def _shipped() -> frozenset[str]:
    """The shipped catalog's factor ids, which no workspace activates."""
    from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog

    return frozenset(FeatureCatalog.load().factor_ids)


class _PlanName(BaseModel):  # type: ignore[misc]
    """A stored research plan's name, to read the plan back through its owner.

    The plan validates only from its JSON (its specs' families are strict enums), so the store
    names the plans and `ResearchFeatureDefinitions.read` reads each.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)
    plan_hash: str


def formula_goldens(
    spec: FactorSpec, specification: FactorFormulaSpecification
) -> list[dict[str, Any]]:
    """Each golden example of a formula's specification as the kernel computes it.

    A golden with no source rows (the one row short of a formula that needs one session) has no
    value to compute: its result is the missing value it expects (IndexError).

    Args:
        spec: The factor's declared recipe.
        specification: Its formula specification, with the golden examples.

    Returns:
        Each example's label, its expected and computed values and whether they agree.
    """
    registry = default_extension_kernel_registry()
    goldens = []
    for example in specification.golden_examples:
        computed = (
            math.nan
            if example.row_count == 0
            else float(registry.compute(golden_example_frame(example), spec).iloc[-1])
        )
        expected = example.expected_value
        within = (
            math.isnan(computed)
            if expected is None
            else math.isfinite(computed)
            and abs(computed - expected)
            <= max(spec.absolute_tolerance, spec.relative_tolerance * abs(expected))
        )
        goldens.append(
            {
                "label": example.label,
                "expected": expected,
                "computed": None if math.isnan(computed) else computed,
                "within_tolerance": within,
            }
        )
    return goldens


class FeatureExtensions:
    """The Host's formula factors: review packets, a person's activation and deactivation."""

    def __init__(
        self,
        workspace: Path,
        clock: Callable[[], datetime],
        *,
        trials: FeatureTrials,
        goals: GoalStore | None,
        rebind: Callable[[], None] | None = None,
    ) -> None:
        """Bind a workspace, its trial ledger and its goals.

        Args:
            workspace: The Host's workspace.
            clock: The Host's clock.
            trials: The feature trials, their builds and plans.
            goals: The goals, whose records count the formulas a goal tried.
            rebind: The Host's rebinding of the workspace's data update to the catalog an
                activation or a deactivation makes.
        """
        self.workspace, self.clock, self.trials, self.goals = workspace, clock, trials, goals
        self.rebind = rebind

    # ------------------------------------------------------------------ the listing

    def listing(self, *, page: int | None = None) -> dict[str, Any]:
        """Every formula factor the workspace's research plans declare, a page at a time.

        One row per plan and factor its edits declare: the formula, its kernel and recipe,
        whether the factor is active, the latest trial of its methodology, and the requests
        its review packet would offer -- the review always, activation or deactivation by the
        packet's own rule (`_next`). Rows with a trial come first, the latest tried first. The
        damaged trial records ride beside as the same explicit refusals returned by
        `FEATURE_TRIALS`, including their detail and next requests.

        Args:
            page: The page, from 1; the first when None.

        Returns:
            `FEATURE_EXTENSIONS`: the page's factors, the page and the page count, the total,
            damaged trial records, unreadable plan refusals and the next page's request while one
            remains.

        Raises:
            ValueError: `feature_extension.extensions_page_out_of_range:<n> of <m>` for a page
                past the last.
        """
        definitions = self.trials.feature_builds.definitions()
        plans: list[ResearchFeaturePlan] = []
        refused_plans: list[dict[str, Any]] = []
        for path in sorted((definitions.store.root / CATEGORY).glob("*.json")):
            # A filename that is not a content identity was ignored by `find_models`; keep that
            # boundary while isolating each actual record read from its neighbours.
            if len(path.stem) != 64:
                continue
            plan_hash = path.stem
            try:
                name = definitions.store.load_model(
                    category=CATEGORY, content_hash=path.stem, model=_PlanName
                )
                if name.plan_hash != path.stem:
                    raise ValueError("feature_research.plan_reference_mismatch")
                plan = definitions.read(path.stem)
            except (OSError, ValueError) as error:
                code = (
                    "feature_extension.plan_unreadable"
                    if isinstance(error, OSError)
                    else public_failure(error, "feature_extension.plan_unreadable")
                )
                words = refusal_words(code)
                detail = words.get("detail") or (
                    f"Saved Feature plan {plan_hash} cannot be read; its input and declared "
                    "factors cannot be trusted."
                )
                refused_plans.append(
                    {
                        "status": "REFUSED",
                        "feature_plan_hash": plan_hash,
                        "failure_code": code,
                        "detail": (
                            f"{detail} To make a new change, use `input list`, choose an input, "
                            "run `feature controls --binding <selected-binding> "
                            "--save-declaration <declaration>`, edit that declaration and run "
                            "`feature plan --file <declaration>`; "
                            "`schema show FEATURE_CATALOG_PLAN` prints its fields."
                        ),
                        "next_requests": {
                            "inputs": {"operation": "RESEARCH_INPUTS"},
                            "workspace": {"operation": "WORKSPACE_SHOW"},
                            "backups": {"operation": "WORKSPACE_BACKUPS"},
                        },
                    }
                )
                continue
            plans.append(plan)
        kept: dict[str, dict[str, FactorSpec]] = {
            plan.plan_hash: {v.factor_id: v.specification for v in plan.candidate.features}
            for plan in plans
        }
        tried: dict[tuple[str, ...], list[FeatureTrial]] = {}
        planned = {plan.plan_hash: plan for plan in plans}
        for trial in self.trials.records():
            for factor_id in trial.feature_factor_ids:
                spec = kept.get(trial.feature_plan_hash, {}).get(factor_id)
                if spec is not None:
                    key = self._trial_key(planned[trial.feature_plan_hash], factor_id)
                    tried.setdefault(key, []).append(trial)
        active = {v.factor_id for v in FeatureExtensionRegistry.read(self.workspace).active}
        shipped = _shipped()
        rows: list[dict[str, Any]] = []
        for plan in plans:
            declared = {
                v.factor_id for v in plan.request.edits if v.operation in {"CREATE", "UPDATE"}
            }
            for factor_id, spec in kept[plan.plan_hash].items():
                if factor_id not in declared or spec.formula_ref not in FORMULA_IDS:
                    continue
                trials = sorted(
                    tried.get(self._trial_key(plan, factor_id), []),
                    key=lambda value: value.requested_at,
                )
                completed = [v for v in trials if v.state == "COMPLETED"]
                recipe = plan.preprocessing_recipes.get(factor_id)
                state = "ACTIVE" if factor_id in active else "NOT_ACTIVE"
                admitted = self._admission(spec, recipe)
                # The contract runs only where it decides an offer.
                passed = (
                    self._contract(spec, recipe, plan.request.input_binding_hash)["status"]
                    if state == "NOT_ACTIVE" and completed and admitted["admitted"]
                    else None
                )
                latest = trials[-1] if trials else None
                rows.append(
                    {
                        "feature_plan_hash": plan.plan_hash,
                        "factor_id": factor_id,
                        "formula": spec.formula,
                        "formula_ref": spec.formula_ref,
                        "preprocessing_recipe": recipe,
                        "state": state,
                        "trials": len(trials),
                        "trial": None
                        if latest is None
                        else {
                            "feature_trial_id": latest.trial_id,
                            "state": latest.state,
                            "outcome": latest.outcome,
                            "requested_at": latest.requested_at.isoformat(),
                        },
                        "next_requests": {
                            "review": {
                                "operation": "FEATURE_REVIEW",
                                "feature_plan_hash": plan.plan_hash,
                                "feature_factor_id": factor_id,
                            },
                            **self._next(
                                state,
                                factor_id,
                                plan.plan_hash,
                                offered=passed == "PASSED" and factor_id not in shipped,
                            ),
                        },
                    }
                )
        # Stable sorts: the tried first, the latest tried first, then by factor and plan.
        rows.sort(key=lambda row: (str(row["factor_id"]), str(row["feature_plan_hash"])))
        rows.sort(
            key=lambda row: "" if row["trial"] is None else str(row["trial"]["requested_at"]),
            reverse=True,
        )
        rows.sort(key=lambda row: row["trial"] is None)
        damaged_trials = self.trials.listing()["damaged"]
        page_count = max(1, math.ceil(len(rows) / _LISTING_PAGE))
        number = 1 if page is None else page
        if not 1 <= number <= page_count:
            raise ValueError(
                f"feature_extension.extensions_page_out_of_range:{number} of {page_count}"
            )
        return {
            "status": "FEATURE_EXTENSIONS",
            "factors": rows[(number - 1) * _LISTING_PAGE : number * _LISTING_PAGE],
            **({"refused_plans": refused_plans} if refused_plans else {}),
            "page": number,
            "page_count": page_count,
            "total": len(rows),
            "damaged": damaged_trials,
            "next_requests": {}
            if number == page_count
            else {"next": {"operation": "FEATURE_EXTENSIONS", "extensions_page": number + 1}},
        }

    def _admission(self, spec: FactorSpec, recipe: str | None) -> dict[str, Any]:
        """Whether the active Panel admits the factor: by its recipe, and a research leaf never."""
        if spec.formula_ref in FORMULA_RESEARCH_IDS:
            # The daily build carries no Sector child yet: a research factor.
            return {
                "admitted": False,
                "reason": f"feature_extension.sector_leaf_research_only:{spec.factor_id}",
            }
        return self._admitted_for_active_panel(recipe)

    # ------------------------------------------------------------------ the packet

    def review(self, feature_plan_hash: str, factor_id: str) -> dict[str, Any]:
        """One formula factor's review packet, what a person reads before activating it.

        Args:
            feature_plan_hash: The research plan that declares it.
            factor_id: The factor.

        Returns:
            `FEATURE_REVIEW`: the declaration and its recipe, the contract (the goldens the
            reference states, computed by the kernel), the identities it adds and moves, the
            environment, the build's coverage and missing share, every trial of this methodology
            with its evidence, the formulas tried in the same goals and in the workspace, the
            activation, and the next request.

        Raises:
            ValueError: `feature_extension.not_a_formula_factor:<id>` when the plan's factor is
                not a formula factor, `feature_extension.factor_absent:<id>` when the plan holds
                no such factor.
        """
        plan = self.trials.feature_builds.definitions().read(feature_plan_hash)
        spec = next(
            (v.specification for v in plan.candidate.features if v.factor_id == factor_id), None
        )
        if spec is None:
            raise ValueError(f"feature_extension.factor_absent:{factor_id}")
        if spec.formula_ref not in FORMULA_IDS:
            raise ValueError(f"feature_extension.not_a_formula_factor:{factor_id}")
        recipe = plan.preprocessing_recipes.get(factor_id)
        contract = self._contract(spec, recipe, plan.request.input_binding_hash)
        registry = default_extension_kernel_registry()
        implementation = registry.implementation_hash(
            spec, core_bundle=desktop_core_feature_bundle()
        )
        methodology = factor_methodology_hash(spec, implementation_hash=implementation)
        trials = self._trials_of(plan, spec)
        admitted = self._admission(spec, recipe)
        activation = next(
            (
                value
                for value in FeatureExtensionRegistry.read(self.workspace).active
                if value.factor_id == factor_id
            ),
            None,
        )
        packet: dict[str, Any] = {
            "status": "FEATURE_REVIEW",
            "factor_id": factor_id,
            "feature_plan_hash": feature_plan_hash,
            "state": "ACTIVE" if activation is not None else "NOT_ACTIVE",
            "declaration": {
                "specification": spec.model_dump(mode="json"),
                "formula": spec.formula,
                "preprocessing_recipe": recipe,
                "preprocessing_rule": formula_controls()["preprocessing_rule"],
            },
            "contract": contract,
            "identity": {
                "implementation_hash": implementation,
                "methodology_hash": methodology,
                "numerical_spec_hash": numerical_spec_hash(spec),
                # A formula factor is data: activation adds a catalog entry and moves no code
                # identity (EX, the formula point).
                "adds": "DAILY_CATALOG_ENTRY",
                "moves": [],
            },
            "environment": recorded_environment(("numpy", "pandas", "scipy")),
            "build": self._build(trials, factor_id),
            "trials": [self._trial(trial, factor_id) for trial in trials],
            "tried": self._tried(trials),
            "active_panel": admitted,
            # What activation does beyond the record: a catalog change recomputes the daily
            # Panel at the next data update, every factor from the history's start.
            "activation_effect": "DAILY_PANEL_REBUILT_AT_NEXT_DATA_UPDATE",
            "activation": None if activation is None else activation.model_dump(mode="json"),
        }
        held = self._held(packet, factor_id)
        # What the packet can claim, one standing from its marks, in what a person reviews.
        packet["standing"] = packet_standing(
            contract=contract,
            trials=packet["trials"],
            active=activation is not None,
            held=held,
        ).model_dump(mode="json")
        packet["packet_hash"] = canonical_hash(
            {key: value for key, value in packet.items() if key not in {"environment"}}
        )
        packet["next_requests"] = self._next(
            packet["state"], factor_id, feature_plan_hash, offered=held is None
        )
        return packet

    def _contract(self, spec: FactorSpec, recipe: str | None, binding: str) -> dict[str, Any]:
        try:
            sessions = len(read_factor_bundle(self.workspace, binding, verify=False).sessions)
        except FileNotFoundError:
            sessions = spec.minimum_observations
        try:
            specification = build_research_formula_specification(
                spec, source_session_count=sessions, preprocessing_recipe=recipe
            )
        except ValueError as error:
            return {
                "status": "FAILED",
                "failure_code": public_failure(error, "feature_extension.contract_failed"),
                "goldens": [],
            }
        goldens = formula_goldens(spec, specification)
        return {
            "status": "PASSED" if all(row["within_tolerance"] for row in goldens) else "FAILED",
            "specification_hash": specification.specification_hash,
            "preprocessing_role": specification.preprocessing_role,
            "goldens": goldens,
        }

    @staticmethod
    def _trial_key(plan: ResearchFeaturePlan, factor_id: str) -> tuple[str, ...]:
        """Bind the executed input, catalog, compute scope and all preprocessing recipes.

        The plan's authored reason and parent-plan reference do not execute. Two declarations
        that build the same columns from the same input therefore share trial evidence, while
        a changed recipe or any changed candidate definition cannot borrow another trial's
        feature or joint Alpha effect.
        """
        return (
            factor_id,
            str(canonical_hash(research_feature_execution_spec(plan))),
        )

    def _trials_of(self, plan: ResearchFeaturePlan, spec: FactorSpec) -> list[FeatureTrial]:
        """Every trial of these executed definitions and input, under any declaration."""
        wanted = self._trial_key(plan, spec.factor_id)
        found = []
        definitions = self.trials.feature_builds.definitions()
        for trial in self.trials.records():
            if spec.factor_id not in trial.feature_factor_ids:
                continue
            tried = definitions.read(trial.feature_plan_hash)
            if self._trial_key(tried, spec.factor_id) == wanted:
                found.append(trial)
        return found

    def _trial(self, trial: FeatureTrial, factor_id: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "feature_trial_id": trial.trial_id,
            "state": trial.state,
            "outcome": trial.outcome,
            "baseline_task_id": trial.baseline_task_id,
            "stopped": trial.stopped,
        }
        if trial.state != "COMPLETED":
            return body
        comparison = cast(dict[str, Any], self.trials.readback(trial.trial_id)["comparison"])
        feature = cast(dict[str, Any], comparison["feature"])
        body["out_of_sample_evidence"] = [
            item
            for item in cast(list[dict[str, Any]], feature["out_of_sample_evidence"])
            if item.get("factor_id") == factor_id
        ]
        body["correlation_with_present_factors"] = [
            pair
            for pair in cast(list[dict[str, Any]], feature["correlation_with_present_factors"])
            if factor_id in {pair.get("left_factor_id"), pair.get("right_factor_id")}
        ]
        alpha = comparison.get("alpha_without_and_with")
        if isinstance(alpha, dict):
            body["alpha_change"] = alpha.get("change")
            # The packet a person activates from says whether the owner compared the two
            # Alpha studies, and gives its words when it did not.
            body["alpha_standing"] = alpha.get("standing")
            owner = alpha.get("owner_comparison")
            if alpha.get("standing") == "NOT_COMPARED" and isinstance(owner, dict):
                body["alpha_refusal"] = {
                    "failure_code": owner.get("failure_code"),
                    "detail": owner.get("detail"),
                }
        return body

    def _build(self, trials: Iterable[FeatureTrial], factor_id: str) -> dict[str, Any] | None:
        """The factor's column in the latest trial's build: its coverage and missing share."""
        for trial in sorted(trials, key=lambda value: value.requested_at, reverse=True):
            task = trial.steps.get("FEATURE_BUILD")
            if task is None:
                continue
            build = self.trials.feature_builds.readback(UUID(task))
            column = next(
                (row for row in build.get("columns", []) if row.get("factor_id") == factor_id),
                None,
            )
            if column is None:
                continue
            cells, available = int(column["cells"]), int(column["available"])
            return {
                "task_id": task,
                "sessions": column["sessions"],
                "listings": column["listings"],
                "cells": cells,
                "available": available,
                "coverage": available / cells if cells else None,
                "missing_share": 1.0 - available / cells if cells else None,
            }
        return None

    def _tried(self, trials: Iterable[FeatureTrial]) -> dict[str, Any]:
        """How many formulas were tried in the goals that tried this one, and in the workspace.

        Every trial is on the ledger; the counts are what a multiple-testing correction reads.
        """
        records = self.trials.records()
        formulas = {trial.trial_id: self._formulas(trial) for trial in records}
        mine = {trial.trial_id for trial in trials}
        goals = []
        for goal_id, trial_ids in self._goal_trials().items():
            if not mine & trial_ids:
                continue
            tried = set().union(*(formulas.get(value, set()) for value in trial_ids))
            goals.append(
                {"goal_id": str(goal_id), "formulas": len(tried), "trials": len(trial_ids)}
            )
        return {
            "goals": goals,
            "workspace_formulas": len(set().union(*formulas.values())) if formulas else 0,
            "workspace_trials": len(records),
        }

    def _formulas(self, trial: FeatureTrial) -> set[str]:
        plan = self.trials.feature_builds.definitions().read(trial.feature_plan_hash)
        return {
            v.specification.formula
            for v in plan.candidate.features
            if v.factor_id in trial.feature_factor_ids
            and v.specification.formula_ref in FORMULA_IDS
        }

    def _goal_trials(self) -> dict[UUID, set[str]]:
        if self.goals is None:
            return {}
        found: dict[UUID, set[str]] = {}
        for goal_id in self.goals.goal_ids():
            trial_ids = {
                str(entry["feature_trial_id"])
                for entry in self.goals.attributed(goal_id)
                if entry.get("operation") == "FEATURE_TRIAL" and entry.get("feature_trial_id")
            }
            if trial_ids:
                found[goal_id] = trial_ids
        return found

    @staticmethod
    def _admitted_for_active_panel(recipe: str | None) -> dict[str, Any]:
        catalog = build_installed_panel_preprocessing_catalog()
        if recipe is None:
            return {"admitted": False, "reason": "feature_extension.preprocessing_recipe_absent"}
        admitted = catalog.capability(recipe).admitted_for_active_panel
        return {
            "admitted": admitted,
            "reason": None
            if admitted
            else f"feature_extension.preprocessing_not_admitted_for_active_panel:{recipe}",
        }

    @staticmethod
    def _held(packet: dict[str, Any], factor_id: str) -> str | None:
        """The code a person's activation is refused with now; None when it is not."""
        if factor_id in _shipped():
            return f"feature_extension.shipped:{factor_id}"
        if packet["contract"]["status"] != "PASSED":
            code = packet["contract"].get("failure_code") or "goldens_outside_tolerance"
            return f"feature_extension.contract_failed:{code}"
        if not any(row["state"] == "COMPLETED" for row in packet["trials"]):
            return f"feature_extension.trial_required:{factor_id}"
        if not packet["active_panel"]["admitted"]:
            return str(packet["active_panel"]["reason"])
        return None

    @staticmethod
    def _next(
        state: str, factor_id: str, feature_plan_hash: str, *, offered: bool
    ) -> dict[str, dict[str, str]]:
        if state == "ACTIVE":
            return {
                "deactivate": {"operation": "FEATURE_DEACTIVATE", "feature_factor_id": factor_id}
            }
        if not offered:
            return {}
        return {
            "activate": {
                "operation": "FEATURE_ACTIVATE",
                "feature_plan_hash": feature_plan_hash,
                "feature_factor_id": factor_id,
            }
        }

    # ------------------------------------------------------------------ a person's acts

    def activate(self, feature_plan_hash: str, factor_id: str) -> dict[str, Any]:
        """A person's activation of a formula factor into this workspace's daily catalog.

        Args:
            feature_plan_hash: The research plan that declares it.
            factor_id: The factor.

        Returns:
            `ACTIVATED` and the record.

        Raises:
            ValueError: `feature_extension.contract_failed:<code>`;
                `feature_extension.trial_required:<id>` when no trial of this methodology
                completed; `feature_extension.preprocessing_not_admitted_for_active_panel:<r>`;
                `feature_extension.shipped:<id>` for a factor the shipped catalog holds.
        """
        if factor_id in _shipped():
            raise ValueError(f"feature_extension.shipped:{factor_id}")
        packet = self.review(feature_plan_hash, factor_id)
        held = self._held(packet, factor_id)
        if held is not None:
            raise ValueError(held)
        completed = [row for row in packet["trials"] if row["state"] == "COMPLETED"]
        record = FeatureActivation(
            factor_id=factor_id,
            specification=FactorSpec.model_validate_json(
                json.dumps(packet["declaration"]["specification"])
            ),
            preprocessing_recipe=packet["declaration"]["preprocessing_recipe"],
            feature_plan_hash=feature_plan_hash,
            trial_id=completed[-1]["feature_trial_id"],
            methodology_hash=packet["identity"]["methodology_hash"],
            packet_hash=packet["packet_hash"],
            activated_at=self.clock(),
        )
        registry = FeatureExtensionRegistry.read(self.workspace)
        others = tuple(value for value in registry.active if value.factor_id != factor_id)
        registry.model_copy(update={"active": (*others, record)}).write(self.workspace)
        if self.rebind is not None:
            self.rebind()
        return {"status": "ACTIVATED", "activation": record.model_dump(mode="json")}

    def deactivate(self, factor_id: str) -> dict[str, Any]:
        """A person's deactivation: the factor leaves this workspace's daily catalog.

        A Panel or study that binds it still reads back; its values stay where they were sealed.

        Args:
            factor_id: The factor.

        Returns:
            `DEACTIVATED`.

        Raises:
            ValueError: `feature_extension.not_active:<id>`.
        """
        registry = FeatureExtensionRegistry.read(self.workspace)
        kept = tuple(value for value in registry.active if value.factor_id != factor_id)
        if len(kept) == len(registry.active):
            raise ValueError(f"feature_extension.not_active:{factor_id}")
        registry.model_copy(update={"active": kept}).write(self.workspace)
        if self.rebind is not None:
            self.rebind()
        return {"status": "DEACTIVATED", "factor_id": factor_id}


__all__ = ["FeatureExtensions", "formula_goldens"]
