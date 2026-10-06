"""Active Feature assertions, reused verbatim without the retired prototype fixture.

The product retired the Feature Catalog's disposable executor, store and result wrapper
(127313b4, "retire isolated workspace and prototype feature executors"); its Planner, Editor,
error and prepared-plan types remain. The prototype's dogfood tests are not a requirement on the
new runtime, and the retired prototype is not restored to satisfy them. These three assertions
are the product's own active-Feature tests as the core-retirement record (playpen-release-e2e)
extracted them: only the
fixture changed -- declaration checks need the catalog and its revision, not a fabricated base,
panel or prototype executor -- and the source pin names this checkout's code.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from alphalattice.foundation.feature_engine.catalog.contracts import FeatureCatalog
from alphalattice.foundation.feature_engine.catalog.service import (
    FeatureCatalogCrudError,
    FeatureCatalogCrudPlanner,
)
from alphalattice.foundation.feature_engine.producers.factors.catalog import (
    default_extension_kernel_registry,
)
from alphalattice.foundation.feature_engine.producers.factors.open_intraday import (
    MEAN_ADJUSTED_RETURN_ID,
)
from alphalattice.foundation.feature_engine.producers.factors.registry import (
    FeatureKernelRegistry,
)
from alphalattice.kernel.quant.factor_enums import FactorFamily

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ID = "zz_mean_adjusted_return_10"
CrudFixture = SimpleNamespace


@pytest.fixture
def crud_fixture():
    catalog = FeatureCatalog.load()
    planner = FeatureCatalogCrudPlanner(default_extension_kernel_registry())
    return CrudFixture(catalog=catalog, revision=planner.revision(catalog))


@pytest.fixture(autouse=True)
def assert_target_source():
    import alphalattice

    assert Path(alphalattice.__file__).resolve() == ROOT / "src/alphalattice/__init__.py"


def _extension_spec(
    fixture: CrudFixture,
    *,
    factor_id: str = EXTENSION_ID,
    window_sessions: int = 10,
) -> object:
    return fixture.catalog.factors[0].model_copy(
        update={
            "factor_id": factor_id,
            "formula_ref": MEAN_ADJUSTED_RETURN_ID,
            "formula": f"mean lagged adjusted log returns over {window_sessions} sessions",
            "window_sessions": window_sessions,
            "lag_sessions": 1,
            "minimum_observations": window_sessions + 2,
            "required_fields": ("provider_adjusted_close",),
            "core_anchor": False,
        }
    )


def test_source_bound_research_edits_preserve_each_revision_and_never_compute(
    crud_fixture, monkeypatch
):
    from alphalattice.foundation.feature_engine.catalog.research import (
        ResearchFeatureChange,
        ResearchFeaturePlan,
        plan_research_feature_change,
        research_feature_controls,
    )

    def no_compute(*_args, **_kwargs):
        raise AssertionError("Definition planning must not run a Feature kernel")

    monkeypatch.setattr(FeatureKernelRegistry, "compute", no_compute)
    fixture = crud_fixture
    catalog, revision = fixture.catalog, fixture.revision
    original = catalog.to_payload()
    plans = []
    spec = _extension_spec(fixture)
    for operation, definition, expected_computed in (
        ("CREATE", spec, (EXTENSION_ID,)),
        ("UPDATE", spec.model_copy(update={"literature_sources": ("research-local note",)}), ()),
        ("UPDATE", spec.model_copy(update={"family": FactorFamily.MOMENTUM}), ()),
        (
            "UPDATE",
            spec.model_copy(update={"window_sessions": 21, "minimum_observations": 23}),
            (EXTENSION_ID,),
        ),
        (
            "RENAME",
            spec.model_copy(update={"factor_id": "research_mean_alias"}),
            ("research_mean_alias",),
        ),
        ("RETIRE", None, ()),
    ):
        factor_id = "research_mean_alias" if operation == "RETIRE" else EXTENSION_ID
        request = ResearchFeatureChange(
            input_binding_hash="a" * 64,
            base_revision_hash=revision.revision_hash,
            parent_plan_hash=None if not plans else plans[-1].plan_hash,
            edits=[{"operation": operation, "factor_id": factor_id, "specification": definition}],
            reason="declared local definition change",
        )
        planned = plan_research_feature_change(
            catalog=catalog,
            base=revision,
            source_panel_snapshot_hash="b" * 64,
            request=request,
        )
        assert planned.work.base_compute_factor_ids == expected_computed
        assert planned.candidate.parent_revision_hash == revision.revision_hash
        assert planned.required_fields == (
            ("provider_adjusted_close",) if expected_computed else ()
        )
        plans.append(planned)
        catalog, revision = (
            FeatureCatalog.from_payload(planned.candidate_payload),
            planned.candidate,
        )
    assert plans[2].delta.research_classification_updated_factor_ids == (EXTENSION_ID,)
    assert plans[4].delta.rename_pairs == ((EXTENSION_ID, "research_mean_alias"),)
    assert fixture.catalog.to_payload() == original
    assert all(ResearchFeaturePlan.model_validate_json(p.model_dump_json()) == p for p in plans)
    bad = plans[0].model_dump(mode="json")
    bad["request"]["input_binding_hash"] = "c" * 64
    with pytest.raises(ValueError, match="plan_binding_invalid"):
        ResearchFeaturePlan.model_validate_json(json.dumps(bad))
    with pytest.raises(FeatureCatalogCrudError, match="exact definition revision"):
        plan_research_feature_change(
            catalog=catalog,
            base=revision,
            source_panel_snapshot_hash="b" * 64,
            request=plans[0].request,
        )
    controls = research_feature_controls()
    assert any(k["formula_ref"] == MEAN_ADJUSTED_RETURN_ID for k in controls["registered_kernels"])
    first = plans[0]

    weakened = spec.model_copy(update={"absolute_tolerance": 1.0})
    with pytest.raises(FeatureCatalogCrudError, match="Validation tolerances"):
        plan_research_feature_change(
            catalog=FeatureCatalog.from_payload(first.candidate_payload),
            base=first.candidate,
            source_panel_snapshot_hash="b" * 64,
            request=ResearchFeatureChange(
                input_binding_hash="a" * 64,
                base_revision_hash=first.candidate.revision_hash,
                edits=[
                    {"operation": "UPDATE", "factor_id": EXTENSION_ID, "specification": weakened}
                ],
                reason="must not silently weaken validation",
            ),
        )


def test_feature_build_retains_raw_contract_and_combines_preprocessing_in_one_task():
    from alphalattice.control.product_host.data_preparation.feature_research import (
        PREPARATION_STAGE,
        STAGE,
        _contract,
    )
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchRequestDocument,
    )

    request = {"operation": "FEATURE_CATALOG_BUILD", "feature_plan_hash": "a" * 64}
    old = PortfolioResearchRequestDocument.model_validate(request).to_operation_request()
    assert old.feature_output is None
    explicit = PortfolioResearchRequestDocument.model_validate(
        {**request, "feature_output": "PREPROCESSED_VALUES"}
    ).to_operation_request()
    assert explicit.feature_output == "PREPROCESSED_VALUES"
    _, old_goal, raw = _contract({"definition_plan_hash": "a" * 64})
    _, new_goal, complete = _contract(
        {"definition_plan_hash": "a" * 64, "purpose": "PREPROCESS_VALUES"}
    )
    assert old_goal.deliverable_kind == "ResearchFeatureMaterialization"
    assert new_goal.deliverable_kind == "ResearchFeaturePreparation"
    assert tuple((v.stage_id, v.dependency_ids) for v in raw.work_items) == ((STAGE, ()),)
    assert tuple((v.stage_id, v.dependency_ids) for v in complete.work_items) == (
        (STAGE, ()),
        (PREPARATION_STAGE, (STAGE,)),
    )
    with pytest.raises(ValueError, match="task_purpose_invalid"):
        _contract({"purpose": "unknown"})
    with pytest.raises(ValueError):
        PortfolioResearchRequestDocument.model_validate(
            {
                "operation": "FEATURE_CATALOG_READBACK",
                "feature_plan_hash": "a" * 64,
                "feature_output": "PREPROCESSED_VALUES",
            }
        ).to_operation_request()


def test_research_formula_value_children_are_bounded_reused_and_tamper_checked(tmp_path):
    from datetime import date

    from alphalattice.control.workspace_runtime.artifacts import ArtifactResolver
    from alphalattice.foundation.feature_engine.catalog.research_values import (
        VALUE_CATEGORY,
        ResearchFormulaValues,
        column_binding_hash,
    )
    from alphalattice.foundation.feature_engine.panels.closure_artifacts import (
        PanelClosureArtifactStore,
    )

    charged = []
    store = PanelClosureArtifactStore(
        ArtifactResolver(tmp_path / "bounded"), capacity=charged.append
    )
    owner = ResearchFormulaValues(store)
    binding = dict(
        input_binding_hash="a" * 64,
        numerical_spec_hash="b" * 64,
        implementation_hash="c" * 64,
        source_projection_hash="d" * 64,
    )
    binding["binding_hash"] = column_binding_hash(**binding)
    binding.update(sessions=(date(2024, 1, 2), date(2024, 1, 3)), listing_ids=("A", "B"))
    values = np.array([[np.nan, 0.0], [1.0, -2.0]])
    first = owner.publish(values=values, **binding)
    before = list(charged)
    assert owner.lookup(first.binding_hash) == first
    assert owner.publish(values=values, **binding) == first and charged == before
    np.testing.assert_array_equal(owner.values(first), values)
    legacy = ResearchFormulaValues(PanelClosureArtifactStore(ArtifactResolver(tmp_path / "legacy")))
    assert legacy.publish(values=values, **binding).values == first.values

    def refuse(_size):
        raise RuntimeError("capacity refused")

    refused = PanelClosureArtifactStore(ArtifactResolver(tmp_path / "refused"), capacity=refuse)
    with pytest.raises(RuntimeError, match="capacity refused"):
        ResearchFormulaValues(refused).publish(values=values, **binding)
    assert not list(refused.root.rglob("*.parquet")) and not list(refused.root.rglob("*.tmp"))
    path = store.physical_path(category=VALUE_CATEGORY, reference=first.values)
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        owner.lookup(first.binding_hash)
