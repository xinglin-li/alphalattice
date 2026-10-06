"""A workspace's daily catalog reads its activated formula factors (EX, part 3c-4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from alphalattice.control.product_host.maintenance.data_update import (
    installed_data_update_binding,
)
from alphalattice.control.product_host.research_authoring.feature_activations import (
    CATALOG_REVISIONS,
    FeatureActivation,
    FeatureExtensionRegistry,
    feature_catalog_for,
    workspace_feature_catalog,
)
from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.producers.factors.formula import (
    formula_specification,
)


def _activation(factor_id: str) -> FeatureActivation:
    shipped = FeatureCatalog.load()
    spec = formula_specification(
        shipped.factors[0].model_copy(
            update={
                "factor_id": factor_id,
                "formula_ref": "factor.formula",
                "formula": "-(close / lag(close, 5) - 1)",
                "lag_sessions": 0,
                "core_anchor": False,
            }
        )
    )
    return FeatureActivation(
        factor_id=factor_id,
        specification=spec,
        preprocessing_recipe="ROBUST_SECTOR_NEUTRAL_Z",
        feature_plan_hash="a" * 64,
        trial_id="b" * 64,
        methodology_hash="c" * 64,
        packet_hash="d" * 64,
        activated_at=datetime.now(UTC),
    )


def test_a_workspace_catalog_is_the_shipped_one_with_its_activations(tmp_path: Path) -> None:
    """requirement (EX): with nothing activated the workspace computes the shipped catalog; an
    activation joins its sorted axis and moves its hash, the data update binds it, and every
    catalog the workspace made resolves by its hash after a deactivation."""

    shipped = FeatureCatalog.load()
    assert workspace_feature_catalog(tmp_path).binding == shipped.binding
    assert installed_data_update_binding(tmp_path) == installed_data_update_binding()

    FeatureExtensionRegistry(active=(_activation("formula_reversal_5"),)).write(tmp_path)
    activated = workspace_feature_catalog(tmp_path)
    assert activated.factor_ids == tuple(sorted((*shipped.factor_ids, "formula_reversal_5")))
    made = activated.binding.catalog_hash
    assert made != shipped.binding.catalog_hash
    assert installed_data_update_binding(tmp_path).feature_catalog_hash == made
    assert (tmp_path / CATALOG_REVISIONS / f"{made}.json").is_file()

    FeatureExtensionRegistry().write(tmp_path)
    assert workspace_feature_catalog(tmp_path).binding == shipped.binding
    kept = feature_catalog_for(tmp_path, made)
    assert kept is not None and kept.factor_ids == activated.factor_ids
    assert feature_catalog_for(tmp_path, shipped.binding.catalog_hash) is not None
    assert feature_catalog_for(tmp_path, "e" * 64) is None


def test_a_feature_review_matches_the_executed_input_catalog_and_preprocessing(
    tmp_path: Path,
) -> None:
    """regression (V509, V526): another declaration of identical work shares trial evidence;
    another recipe, input revision or joint feature definition cannot borrow its effect or
    build, and neither the packet nor the listing offers activation from that other trial.
    """
    import json

    from alphalattice.control.product_host.composition.feature_trials import FeatureTrial
    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        FeatureExtensions,
    )
    from alphalattice.control.product_host.research_authoring.feature_research import (
        CATEGORY,
        ResearchFeatureDefinitions,
    )
    from alphalattice.foundation.feature_engine.catalog.research import (
        ResearchFeatureChange,
        plan_research_feature_change,
    )
    from alphalattice.foundation.feature_engine.catalog.service import FeatureCatalogCrudPlanner
    from alphalattice.foundation.feature_engine.producers.factors.catalog import (
        default_extension_kernel_registry,
    )
    from alphalattice.foundation.feature_engine.producers.factors.formula import formula_controls

    factor = "formula_reversal_5"
    spec = _activation(factor).specification
    catalog = FeatureCatalog.load()
    base = FeatureCatalogCrudPlanner(default_extension_kernel_registry()).revision(catalog)
    definitions = ResearchFeatureDefinitions(tmp_path)
    recipe = "ROBUST_SECTOR_NEUTRAL_Z"
    alternative = next(
        value for value in formula_controls()["preprocessing_recipes"] if value != recipe
    )

    def declare(*, reason="Original trial", preprocessing=recipe, binding="a" * 64, joint=False):
        edits = [
            {
                "operation": "CREATE",
                "factor_id": factor,
                "specification": spec.model_dump(mode="json"),
                "preprocessing_recipe": preprocessing,
            }
        ]
        if joint:
            other = _activation("formula_other").specification
            edits.append(
                {
                    "operation": "CREATE",
                    "factor_id": other.factor_id,
                    "specification": other.model_dump(mode="json"),
                    "preprocessing_recipe": recipe,
                }
            )
        request = ResearchFeatureChange.model_validate_json(
            json.dumps(
                {
                    "input_binding_hash": binding,
                    "base_revision_hash": base.revision_hash,
                    "edits": edits,
                    "reason": reason,
                }
            )
        )
        plan = plan_research_feature_change(
            catalog=catalog, base=base, source_panel_snapshot_hash="b" * 64, request=request
        )
        definitions.store.publish_json(
            category=CATEGORY, content_hash=plan.plan_hash, payload=plan.model_dump(mode="json")
        )
        return plan

    original = declare()
    identical = declare(reason="The same work described again")
    different = [declare(preprocessing=alternative), declare(binding="c" * 64), declare(joint=True)]
    assert identical.plan_hash != original.plan_hash
    trial = FeatureTrial(
        trial_id="d" * 64,
        feature_plan_hash=original.plan_hash,
        baseline_task_id="baseline",
        baseline_alpha_task_id="baseline",
        caller="HUMAN",
        requested_at=datetime(2026, 10, 2, tzinfo=UTC),
        state="COMPLETED",
        outcome="COMPARED",
        feature_factor_ids=(factor,),
        steps={"FEATURE_BUILD": "build"},
    )

    class Builds:
        def definitions(self):
            return definitions

        def readback(self, task_id):
            assert str(task_id) == "00000000-0000-4000-8000-000000000001"
            return {
                "columns": [
                    {
                        "factor_id": factor,
                        "sessions": 10,
                        "listings": 2,
                        "cells": 20,
                        "available": 18,
                    }
                ]
            }

    trial.steps["FEATURE_BUILD"] = "00000000-0000-4000-8000-000000000001"

    class Trials:
        feature_builds = Builds()

        def records(self):
            return (trial,)

        def readback(self, trial_id):
            assert trial_id == trial.trial_id
            return {
                "comparison": {
                    "feature": {
                        "out_of_sample_evidence": [],
                        "correlation_with_present_factors": [],
                    },
                    "alpha_without_and_with": {
                        "standing": "COMPARED",
                        "change": {"mean_rank_ic": 0.125},
                    },
                }
            }

        def listing(self):
            return {"damaged": []}

    ledger = Trials()
    extensions = FeatureExtensions(tmp_path, lambda: trial.requested_at, trials=ledger, goals=None)
    for plan in (original, identical):
        packet = extensions.review(plan.plan_hash, factor)
        assert [row["feature_trial_id"] for row in packet["trials"]] == [trial.trial_id]
        assert packet["trials"][0]["alpha_change"] == {"mean_rank_ic": 0.125}
        assert packet["build"]["coverage"] == 0.9
        assert packet["next_requests"]["activate"]["feature_plan_hash"] == plan.plan_hash
    for plan in different:
        packet = extensions.review(plan.plan_hash, factor)
        assert packet["trials"] == [] and packet["build"] is None
        assert packet["standing"]["activation"] == "HELD"
        assert "activate" not in packet["next_requests"]
    listing = {
        row["feature_plan_hash"]: row
        for row in extensions.listing()["factors"]
        if row["factor_id"] == factor
    }
    assert listing[identical.plan_hash]["trial"]["feature_trial_id"] == trial.trial_id
    for plan in different:
        assert listing[plan.plan_hash]["trial"] is None
        assert "activate" not in listing[plan.plan_hash]["next_requests"]


def test_feature_trial_keys_cover_every_executed_definition_field() -> None:
    """contract: generated field mutations cover the execution projection, not named recipes."""
    from types import SimpleNamespace

    from alphalattice.control.product_host.research_authoring.feature_extensions import (
        FeatureExtensions,
    )
    from alphalattice.foundation.feature_engine.catalog.crud_contracts import (
        FeatureCatalogUpdatePlan,
    )
    from alphalattice.foundation.feature_engine.catalog.research import (
        research_feature_execution_spec,
    )

    class Work:
        def __init__(self, values):
            self.values = values

        def model_dump(self, *, mode, exclude):
            return {key: value for key, value in self.values.items() if key not in exclude}

    def plan(**changes):
        values = {
            "request": SimpleNamespace(input_binding_hash="input"),
            "source_panel_snapshot_hash": "panel",
            "candidate": SimpleNamespace(revision_hash="catalog"),
            "work": Work({name: name for name in FeatureCatalogUpdatePlan.model_fields}),
            "required_fields": ("close",),
            "preprocessing_recipes": {"formula": "recipe"},
        }
        return SimpleNamespace(**(values | changes))

    original = plan()
    key = FeatureExtensions._trial_key(original, "formula")
    assert FeatureExtensions._trial_key(plan(), "formula") == key
    for name in research_feature_execution_spec(original):
        if name == "input_binding_hash":
            changed = plan(request=SimpleNamespace(input_binding_hash="other"))
        elif name == "candidate_revision_hash":
            changed = plan(candidate=SimpleNamespace(revision_hash="other"))
        elif name == "work":
            for field in research_feature_execution_spec(original)["work"]:
                changed = plan(work=Work(original.work.values | {field: "other"}))
                assert FeatureExtensions._trial_key(changed, "formula") != key, field
            continue
        else:
            value = {"formula": "other"} if name == "preprocessing_recipes" else ("other",)
            changed = plan(**{name: value})
        assert FeatureExtensions._trial_key(changed, "formula") != key, name
    for name in ("threads", "workers", "batch", "cache"):
        assert FeatureExtensions._trial_key(plan(**{name: 2}), "formula") == key


def test_feature_build_reuses_execution_but_not_changed_source_axes(tmp_path: Path) -> None:
    """regression: a plan's reason does not rebuild; recipes and complete axes cannot alias."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from uuid import UUID

    from alphalattice.control.product_host.data_preparation.feature_research import (
        TASK_KIND,
        ResearchFeatureBuildApplication,
        implementation_hash,
    )
    from alphalattice.control.product_host.research_authoring.feature_materialization import (
        feature_source_projection_hash,
    )
    from alphalattice.control.task_control.contracts import TaskLifecycle

    class Work:
        def model_dump(self, **kwargs):
            return {"base_compute_factor_ids": ["formula"], "panel_compute_factor_ids": ["formula"]}

    def definition(recipe):
        return SimpleNamespace(
            request=SimpleNamespace(input_binding_hash="a" * 64),
            source_panel_snapshot_hash="b" * 64,
            candidate=SimpleNamespace(revision_hash="c" * 64),
            work=Work(),
            required_fields=("close",),
            preprocessing_recipes={"formula": recipe},
        )

    plans = {
        "1" * 64: definition("recipe-a"),
        "2" * 64: definition("recipe-a"),
        "3" * 64: definition("recipe-b"),
    }
    payload = {
        "definition_plan_hash": "1" * 64,
        "input_binding_hash": "a" * 64,
        "sessions_hash": "d" * 64,
        "listing_ids_hash": "e" * 64,
        "implementation_hash": implementation_hash(preparation=True),
        "source_projection_hash": feature_source_projection_hash(),
        "caller": "HUMAN",
        "purpose": "PREPROCESS_VALUES",
    }

    class Registry:
        task = None

        def tasks(self):
            return () if self.task is None else (self.task,)

        def admit(self, *, input_envelope, goal, plan, observed_at):
            self.task = SimpleNamespace(
                task_kind=TASK_KIND,
                input=input_envelope,
                goal=goal,
                plan=plan,
                task_id=UUID("00000000-0000-4000-8000-000000000001"),
                lifecycle=TaskLifecycle.SUCCEEDED,
            )
            return SimpleNamespace(record=self.task)

    registry = Registry()

    class Owner(ResearchFeatureBuildApplication):
        def definitions(self, **kwargs):
            return SimpleNamespace(read=lambda name: plans[name])

        def readback(self, task_id):
            assert task_id == task.task_id
            return {"status": "BUILT", "columns": [{"available": 18}]}

    class Dispatcher:
        calls = 0

        def submit(self, command):
            self.calls += 1
            return SimpleNamespace(
                disposition="ADMITTED",
                task_id=task.task_id,
                lifecycle="QUEUED",
                refusal_detail=None,
            )

    owner = Owner(
        SimpleNamespace(
            workspace=tmp_path,
            task_control_registry=registry,
            mutation_gate=SimpleNamespace(hold=nullcontext),
        ),
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )
    owner.admit(payload)
    task = registry.task
    dispatcher = Dispatcher()
    for change in (
        {},
        {"definition_plan_hash": "2" * 64, "caller": "EXTERNAL_AUTOMATION"},
        *({name: 2} for name in ("threads", "workers", "batch", "cache")),
    ):
        result = owner._submit(payload | change, dispatcher)
        assert result["status"] == "REUSED_EXACT" and result["columns"] == [{"available": 18}]
        assert (
            result["kernel_calls_this_request"] == result["preprocessing_calls_this_request"] == 0
        )
    assert dispatcher.calls == 0
    for name in ("input_binding_hash", "sessions_hash", "listing_ids_hash"):
        assert owner._submit(payload | {name: "9" * 64}, dispatcher)["status"] == "ADMITTED"
    assert (
        owner._submit(payload | {"definition_plan_hash": "3" * 64}, dispatcher)["status"]
        == "ADMITTED"
    )
    assert dispatcher.calls == 4


def test_feature_trial_reopens_equivalent_declarations_on_one_baseline(tmp_path: Path) -> None:
    """regression: the trial ledger reuses complete work, not a declaration's prose hash."""
    from types import SimpleNamespace
    from uuid import UUID

    from alphalattice.control.product_host.composition.feature_trials import FeatureTrials

    class Work:
        def model_dump(self, **kwargs):
            return {"base_compute_factor_ids": ["formula"]}

    def definition(recipe):
        return SimpleNamespace(
            request=SimpleNamespace(input_binding_hash="a" * 64),
            source_panel_snapshot_hash="b" * 64,
            candidate=SimpleNamespace(revision_hash="c" * 64),
            work=Work(),
            required_fields=("close",),
            preprocessing_recipes={"formula": recipe},
        )

    plans = {
        "1" * 64: definition("recipe-a"),
        "2" * 64: definition("recipe-a"),
        "3" * 64: definition("recipe-b"),
    }
    baseline = UUID("00000000-0000-4000-8000-000000000001")

    class Ledger(FeatureTrials):
        def _baseline(self, task_id):
            return task_id, None

        def _plan(self, task_id):
            return SimpleNamespace(binding=SimpleNamespace(binding_hash="a" * 64))

        def _advance(self, trial):
            # No calculation is installed in this synthetic ledger; a completed record's
            # state, timestamp and original declaration stay exactly as saved.
            assert trial.trial_id in {item.trial_id for item in self.records()}

        def readback(self, trial_id):
            return self._read(trial_id).model_dump(mode="json")

    ledger = Ledger(
        workspace=tmp_path,
        experiments=None,
        dispatcher=None,
        feature_builds=SimpleNamespace(
            definitions=lambda: SimpleNamespace(read=lambda h: plans[h])
        ),
        clock=lambda: datetime(2026, 10, 2, tzinfo=UTC),
    )
    original = ledger.start("1" * 64, baseline, caller="HUMAN")
    assert ledger.start("2" * 64, baseline, caller="EXTERNAL_AUTOMATION") == original
    assert len(ledger.records()) == 1
    assert ledger.start("3" * 64, baseline, caller="HUMAN")["trial_id"] != original["trial_id"]
    another = UUID("00000000-0000-4000-8000-000000000002")
    assert ledger.start("1" * 64, another, caller="HUMAN")["trial_id"] != original["trial_id"]
    assert len(ledger.records()) == 3
