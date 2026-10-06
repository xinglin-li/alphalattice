"""A model's search axes, declaration and contract (EX, its design's sections 2 and 5)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from alphalattice.capabilities.alpha_modeling.catalog import build_installed_alpha_model_catalog
from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaModelPredictionResult,
    AlphaModelSearchDomainEnvelope,
)
from alphalattice.capabilities.alpha_modeling.declaration import (
    AlphaModelDeclaration,
    installed_declaration,
)
from alphalattice.capabilities.alpha_modeling.model_contract import (
    contract_findings,
    locked_distributions,
)
from alphalattice.capabilities.alpha_modeling.search_axes import SearchSpace
from alphalattice.investment.alpha_research.experiments.mandate import (
    build_installed_alpha_model_capability_mandate,
)

ROOT = Path(__file__).resolve().parents[2]
LOCKED = locked_distributions(ROOT / "uv.lock")


def _installed() -> list[tuple[Any, AlphaModelSearchDomainEnvelope]]:
    domains = {
        value.adapter_id: value
        for value in build_installed_alpha_model_capability_mandate().ordered_search_domains
    }
    return [
        (adapter, domains[adapter.adapter_id])
        for adapter in build_installed_alpha_model_catalog().adapters
    ]


@pytest.mark.parametrize("model_id", ["regularized_linear", "dynamic_panel_lightgbm"])
def test_an_installed_model_passes_its_contract(model_id: str) -> None:
    """requirement (EX): every adapter passes the contract, the installed ones included; each
    installed model declares its axes beside its adapter, their defaults its reference recipe."""

    ((adapter, domain),) = [v for v in _installed() if v[0].adapter_id == model_id]
    findings = contract_findings(adapter, installed_declaration(model_id), domain, locked=LOCKED)
    assert [(v.check, v.code) for v in findings] == [
        (check, None)
        for check in (
            "route",
            "search_axes",
            "fit_protocol",
            "determinism",
            "prediction_rows",
            "state",
            "imports",
        )
    ]


def test_the_installed_catalog_declares_every_model() -> None:
    """requirement (EX): every installed adapter has a declaration, and the Dynamic Panel
    LightGBM's reference recipe is the product's current one (IW184's point, first seed)."""

    from alphalattice.investment.alpha_research.scores.product_recipe import (
        PRODUCT_ESTIMATOR_POINT,
        PRODUCT_SEEDS,
    )

    assert {installed_declaration(a.adapter_id).model_id for a, _ in _installed()} == {
        "regularized_linear",
        "dynamic_panel_lightgbm",
    }
    point = PRODUCT_ESTIMATOR_POINT.resolve(seed=PRODUCT_SEEDS[0])
    recipe = installed_declaration("dynamic_panel_lightgbm").recipe
    assert recipe == {name: getattr(point, name) for name in recipe}


_AXIS = {"name": "alpha", "kind": "float", "low": 0.1, "high": 100.0, "default": 10.0}


@pytest.mark.parametrize(
    ("axes", "code"),
    [
        ([{**_AXIS, "default": 1000.0}], "search_axes.default_outside_axis:alpha"),
        ([{**_AXIS, "low": 0.0, "log": True}], "search_axes.log_bounds_invalid:alpha"),
        ([{**_AXIS, "low": 100.0}], "search_axes.bounds_invalid:alpha"),
        (
            [{"name": "family", "kind": "categorical", "default": "ridge"}],
            "search_axes.choices_required:family",
        ),
        ([{**_AXIS, "kind": "int", "low": 0.5}], "search_axes.value_not_integral:alpha"),
        ([_AXIS, _AXIS], "search_axes.axis_duplicated:alpha"),
        (
            [{**_AXIS, "when": {"axis": "family", "in": ["ridge"]}}],
            "search_axes.condition_invalid:alpha",
        ),
        (
            [
                {"name": "family", "kind": "categorical", "choices": ["ridge"], "default": "ridge"},
                {**_AXIS, "when": {"axis": "family", "in": ["lasso"]}},
            ],
            "search_axes.condition_choice_unknown:alpha",
        ),
        (
            [
                {"name": "family", "kind": "categorical", "choices": ["ridge"], "default": "ridge"},
                {
                    "name": "solver",
                    "kind": "categorical",
                    "choices": ["a"],
                    "default": "a",
                    "when": {"axis": "family", "in": ["ridge"]},
                },
                {**_AXIS, "when": {"axis": "solver", "in": ["a"]}},
            ],
            "search_axes.condition_invalid:alpha",
        ),
    ],
)
def test_malformed_axes_are_refused_by_name(axes: list[dict[str, Any]], code: str) -> None:
    """requirement (EX): the axes' format is checked as declared: a default inside its axis,
    positive bounds for a log axis, choices for a categorical axis, and one level of `when`
    naming an earlier categorical axis's choices."""

    with pytest.raises(ValidationError, match=re.escape(code)):
        SearchSpace.model_validate({"axes": axes})


def test_a_declaration_is_held_to_its_reference_recipe() -> None:
    """requirement (EX, the user): an axis's default is the reference recipe's value, an axis
    that does not exist under the recipe leaves its parameter null, and the probes of a
    conditional axis state it under a parent choice it exists for."""

    document = installed_declaration("regularized_linear").model_dump(mode="json", by_alias=True)
    moved = {**document, "recipe": {**document["recipe"], "alpha": 1.0}}
    with pytest.raises(
        ValidationError, match=re.escape("model_declaration.default_not_the_recipe:alpha")
    ):
        AlphaModelDeclaration.model_validate(moved)
    valued = {**document, "recipe": {**document["recipe"], "l1_ratio": 0.5}}
    with pytest.raises(
        ValidationError, match=re.escape("model_declaration.inactive_axis_valued:l1_ratio")
    ):
        AlphaModelDeclaration.model_validate(valued)
    space = installed_declaration("regularized_linear").search
    families = {point["family"]: set(point) for point in space.probes()}
    assert families == {
        "ridge": {"family", "alpha"},
        "lasso": {"family", "alpha_max_multiplier"},
        "elastic_net": {"family", "alpha_max_multiplier", "l1_ratio"},
    }


@dataclass
class _Tampered:
    """The installed linear adapter with one behaviour changed."""

    inner: Any
    change: str

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def fit(self, **fields: Any) -> Any:
        result = self.inner.fit(**fields)
        if self.change != "state":
            return result
        from dataclasses import replace

        from alphalattice.capabilities.alpha_modeling.contracts import AlphaModelStateProjection

        empty = AlphaModelStateProjection.create(
            adapter_id=result.state_projection.adapter_id,
            model_family_id="ridge",
            state_kind="LINEAR",
            state_schema_id=result.state_projection.state_schema_id,
            payload={},
        )
        return replace(result, state_projection=empty)

    def predict(self, *, estimator: Any, inputs: Any) -> AlphaModelPredictionResult:
        result = self.inner.predict(estimator=estimator, inputs=inputs)
        values = np.array(result.predictions)
        if self.change == "state":
            return result
        if self.change == "rows":
            values = values - values.mean()  # a row's prediction reads every row
        else:
            values = values + np.random.default_rng().standard_normal(len(values)) * 1e-9
        values.setflags(write=False)
        return AlphaModelPredictionResult(predictions=values)


@pytest.mark.parametrize(
    ("change", "check", "code"),
    [
        ("rows", "prediction_rows", "model_contract.prediction_reads_other_rows"),
        ("noise", "determinism", "model_contract.predictions_not_deterministic"),
        # The state check names each field a LINEAR state lacks (V350).
        (
            "state",
            "state",
            "model_contract.linear_state_incomplete:coefficient_hex,intercept_hex,"
            "coefficient_l2_norm,coefficient_max_abs,nonzero_support_count",
        ),
    ],
)
def test_the_contract_names_what_it_found(change: str, check: str, code: str) -> None:
    """tamper (EX, V342): a prediction that reads other rows, one that moves between two fits
    under one seed, and a LINEAR state without its coefficients each fail the contract by name."""

    ((adapter, domain),) = [v for v in _installed() if v[0].adapter_id == "regularized_linear"]
    findings = contract_findings(
        _Tampered(adapter, change),
        installed_declaration("regularized_linear"),
        domain,
        locked=LOCKED,
    )
    assert {v.check: v.code for v in findings}[check] == code


def test_an_axis_outside_the_domain_and_an_unlocked_import_are_refused() -> None:
    """tamper (EX, the user): an axis wider than the adapter's domain is refused by the
    adapter's own refusal, and a library the lock does not hold is a new dependency the
    contract names, so a person approves it first."""

    ((adapter, domain),) = [v for v in _installed() if v[0].adapter_id == "regularized_linear"]
    document = installed_declaration("regularized_linear").model_dump(mode="json", by_alias=True)
    document["search"]["axes"][1]["high"] = 1000.0
    wide = AlphaModelDeclaration.model_validate(document)
    findings = {
        v.check: v.code
        for v in contract_findings(adapter, wide, domain, locked=LOCKED - {"scikit-learn"})
    }
    assert findings["search_axes"].startswith("model_contract.axis_point_refused:")
    assert findings["search_axes"].endswith("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
    assert findings["imports"] == "model_contract.import_outside_lock:sklearn"
