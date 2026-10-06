"""`model scaffold`: an agent's model's three files, written from its declaration (EX).

The adapter under `extensions/`, whose `fit` and `predict` the agent writes; the declaration
beside it; and its contract test. A declared library the lock does not hold is a new
dependency a person approves first, and nothing is written until then.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from alphalattice.kernel.shared_kernel.project_layout import resolve_playpen_root

from .declaration import read_declaration
from .model_contract import locked_distributions

_SOURCE = Path("src/alphalattice/capabilities/alpha_modeling/extensions")
_TESTS = Path("tests/alpha_research/extensions")

_ADAPTER_TEXT = '''"""The {family} model `{model_id}`, an agent's Alpha model (EX).

Its declaration is `{model_id}.model.yaml`; `model check {model_id}` runs its contract.
"""

from __future__ import annotations

from alphalattice.capabilities.alpha_modeling.contracts import (
    AlphaEstimatorContent,
    AlphaModelFitResult,
    AlphaModelPredictionResult,
    AlphaModelRecipeEnvelope,
    BoundAlphaModelFitInput,
    BoundAlphaPredictionInput,
    BoundAlphaTrainingInput,
)
from alphalattice.capabilities.alpha_modeling.extension import DeclaredModelAdapter


class {camel}Adapter(DeclaredModelAdapter):
    """`{model_id}`: its fit and its prediction; the rest is its declaration's."""

    def fit(
        self,
        *,
        recipe: AlphaModelRecipeEnvelope,
        inputs: BoundAlphaTrainingInput,
        fit_plan: BoundAlphaModelFitInput | None = None,
    ) -> AlphaModelFitResult:
        """Fit on the training rows alone, deterministically under the recipe's seed.

        Args:
            recipe: The recipe (`self.validate_recipe(recipe)` gives its parameters).
            inputs: The training rows.
            fit_plan: The plan the runtime hands the declared protocol.

        Returns:
            The estimator (`AlphaEstimatorContent.create` with `self.content_format_id`); its
            state (`AlphaModelStateProjection.create`): LINEAR for a model linear in the
            features, its payload `coefficient_hex`, `intercept_hex`, `coefficient_l2_norm`,
            `coefficient_max_abs` and `nonzero_support_count`; TREE for a tree ensemble,
            `model_text_hash`, `best_iteration`, `feature_gain_hex` and `top_feature_gain_share`
            (`model_contract.STATE_CONTRACT` gives each field's type and shape); or a kind of
            its own whose payload a study keeps by its hash; and the training error.

        Raises:
            ValueError: Until written.
        """
        del recipe, inputs, fit_plan
        raise ValueError("model_extension.fit_not_written:{model_id}")

    def predict(
        self, *, estimator: AlphaEstimatorContent, inputs: BoundAlphaPredictionInput
    ) -> AlphaModelPredictionResult:
        """Predict each row from that row and the estimator alone.

        Args:
            estimator: What `fit` returned.
            inputs: The prediction rows.

        Returns:
            One finite float64 per row, read only.

        Raises:
            ValueError: Until written.
        """
        del estimator, inputs
        raise ValueError("model_extension.predict_not_written:{model_id}")


ADAPTER = {camel}Adapter
'''

_PACKAGE_TEXT = '"""The Alpha models agents added (EX); a workspace installs the activated."""'

_TEST_TEXT = '''"""`{model_id}`'s contract (EX): it passes before a person activates it."""

from alphalattice.capabilities.alpha_modeling.model_contract import check_model


def test_{model_id}_passes_its_contract() -> None:
    """requirement (EX): the model's contract passes, each check by name."""

    answer = check_model("{model_id}")
    assert answer["status"] == "PASSED", answer["findings"]
'''


def declaration_template() -> str:
    """A model's declaration to edit, which `model scaffold --save-declaration` writes (V413).

    The contract's fields, each with what it means and whether it is required, around the
    installed regularized linear model's own declaration as the worked example, its id renamed:
    the first step of a model of one's own starts from what the contract takes, never a guess.

    Returns:
        The declaration's YAML text.
    """
    import yaml  # type: ignore[import-untyped]

    from .declaration import AlphaModelDeclaration, installed_declaration

    example = installed_declaration("regularized_linear").model_dump(mode="json")
    example = {name: value for name, value in example.items() if value is not None}
    example["model_id"] = "my_model"
    lines = [
        "# A model's declaration to edit; `model scaffold --file <this file>` then writes the",
        "# model's adapter, declaration and contract test from it. The values are the installed",
        "# regularized_linear model's, a worked example: replace each with your model's.",
        "#",
    ]
    for name, field in AlphaModelDeclaration.model_fields.items():
        need = "required" if field.is_required() else "optional"
        lines.append(f"# {name} ({need}): {field.description}")
    body: str = yaml.safe_dump(example, sort_keys=False, allow_unicode=True)
    return "\n".join(lines) + "\n" + body


def scaffold_model(declaration: Path, *, root: Path | None = None) -> dict[str, Any]:
    """`model scaffold`: write an extension's three files from its declaration.

    Args:
        declaration: The declaration's YAML file.
        root: The checkout; the one this module is in when omitted.

    Returns:
        `SCAFFOLDED`, the files written, and what to do next.

    Raises:
        ValueError: `model_extension.library_outside_lock:<names>` for a declared library the lock
            does not hold (a new dependency: a person approves it first, and nothing is
            written); `model_extension.entry_exists:<model_id>` for a model already present.
    """
    checkout = root or resolve_playpen_root(Path(__file__))
    if checkout.name == "_runtime" and checkout.parent.name == "alphalattice":
        raise ValueError("model_extension.editable_checkout_required")
    declared = read_declaration(declaration)
    locked = locked_distributions(checkout / "uv.lock")
    outside = sorted(
        value for value in declared.libraries if re.sub(r"[-_.]+", "-", value).lower() not in locked
    )
    if outside:
        raise ValueError("model_extension.library_outside_lock:" + ",".join(outside))
    model_id = declared.model_id
    source, tests = checkout / _SOURCE, checkout / _TESTS
    files = {
        source / f"{model_id}.py": _ADAPTER_TEXT.format(
            model_id=model_id,
            family=declared.family,
            camel="".join(part.capitalize() for part in model_id.split("_")),
        ),
        source / f"{model_id}.model.yaml": declaration.read_text(encoding="utf-8"),
        tests / f"test_{model_id}.py": _TEST_TEXT.format(model_id=model_id),
    }
    from .catalog import build_installed_alpha_model_catalog

    installed = build_installed_alpha_model_catalog().adapter_ids
    if model_id in installed or any(path.exists() for path in files):
        raise ValueError(f"model_extension.entry_exists:{model_id}")
    # Both packages start with the first extension: a package owns something (the
    # structural guard's empty-shell rule).
    for package, text in ((source, _PACKAGE_TEXT), (tests, "")):
        if not (package / "__init__.py").exists():
            package.mkdir(parents=True, exist_ok=True)
            (package / "__init__.py").write_text(
                f"{text}\n" if text else "", encoding="utf-8", newline="\n"
            )
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return {
        "status": "SCAFFOLDED",
        "model_id": model_id,
        "files": [path.relative_to(checkout).as_posix() for path in files],
        "next_action": "WRITE_FIT_AND_PREDICT_THEN_MODEL_CHECK",
    }


__all__ = ["scaffold_model"]
