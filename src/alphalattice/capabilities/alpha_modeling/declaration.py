"""A model's declaration: its reference recipe, search axes and fit protocol (EX, section 2).

What `model scaffold` will read and `model check` holds. An installed model declares itself in
`adapters/<model_id>.model.yaml`, beside its adapter. This wave a declaration enters no identity:
the recipe a study runs and the domain a mandate seals stay their own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contracts import AlphaModelFitProtocol
from .search_axes import AxisValue, SearchSpace

type ParameterType = Literal["int", "float", "str", "bool"]

_AXIS_KINDS: dict[str, str] = {"int": "int", "float": "float", "str": "categorical", "bool": "bool"}
_ADAPTERS = Path(__file__).resolve().parent / "adapters"


class _Contract(BaseModel):  # type: ignore[misc]
    model_config = ConfigDict(extra="forbid", frozen=True)


def typed_value(value: object, kind: ParameterType) -> bool:
    """Whether a parameter value is of its declared type (a bool is no number).

    Args:
        value: The value.
        kind: The declared type.

    Returns:
        Whether it is.
    """
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if isinstance(value, bool):
        return False
    return isinstance(value, int) if kind == "int" else isinstance(value, int | float)


class AlphaModelDeclaration(_Contract):
    """One model: the recipe's parameters, their reference values, the axes and the protocol."""

    model_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="The adapter's id.")
    family: str = Field(min_length=1, description="The model family, in words.")
    fit_protocol: AlphaModelFitProtocol = Field(
        description="How the reference recipe fits: its whole training surface "
        "(`DIRECT_FIT`), or a tuning partition first (`NESTED_EARLY_STOPPING_REFIT`)."
    )
    parameters: dict[str, ParameterType] = Field(
        min_length=1, description="The recipe's parameters and their types, in recipe order."
    )
    recipe: dict[str, AxisValue | None] = Field(
        description="The reference recipe: each parameter's current value; null where a "
        "conditional axis does not exist under the recipe."
    )
    search: SearchSpace = Field(description="The ordered axes a search may propose from.")
    libraries: tuple[str, ...] = Field(
        default=(),
        description="The distributions its code imports beyond the standard library; each "
        "must be held by the lock, or it is a new dependency a person approves first.",
    )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _consistent(self) -> Self:
        if list(self.recipe) != list(self.parameters):
            raise ValueError("model_declaration.recipe_parameters_differ")
        for name, kind in self.parameters.items():
            value = self.recipe[name]
            if value is not None and not typed_value(value, kind):
                raise ValueError(f"model_declaration.recipe_value_mistyped:{name}")
        point: dict[str, AxisValue] = {}
        for axis in self.search.axes:
            held = self.recipe.get(axis.name)
            if held is not None:
                point[axis.name] = held
        for axis in self.search.axes:
            if axis.name not in self.parameters:
                raise ValueError(f"model_declaration.axis_names_no_parameter:{axis.name}")
            if _AXIS_KINDS[self.parameters[axis.name]] != axis.kind:
                raise ValueError(f"model_declaration.axis_kind_differs:{axis.name}")
            value = self.recipe[axis.name]
            if not self.search.holds(axis, point):
                if value is not None:
                    raise ValueError(f"model_declaration.inactive_axis_valued:{axis.name}")
            # The default is the reference recipe's value (the user, 2026-09-30).
            elif type(value) is not type(axis.default) or value != axis.default:
                raise ValueError(f"model_declaration.default_not_the_recipe:{axis.name}")
        return self

    def recipe_at(self, point: dict[str, AxisValue]) -> dict[str, Any]:
        """The recipe a search point states.

        Its axes' values, the reference recipe's values for parameters no axis governs, and
        null for an axis the point leaves inactive.

        Args:
            point: A complete point of the search space (`SearchSpace.complete`).

        Returns:
            The parameters, in recipe order.
        """
        axes = {axis.name for axis in self.search.axes}
        return {
            name: point.get(name) if name in axes else self.recipe[name] for name in self.parameters
        }


def read_declaration(path: Path) -> AlphaModelDeclaration:
    """Read one declaration.

    Args:
        path: The YAML file.

    Returns:
        The declaration.

    Raises:
        ValueError: `model_declaration.unreadable`, or the declaration's own refusal.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError("model_declaration.unreadable") from error
    try:
        declaration: AlphaModelDeclaration = AlphaModelDeclaration.model_validate(document)
        return declaration
    except ValidationError as error:
        raise ValueError(
            next(
                (
                    str(item["ctx"]["error"])
                    for item in error.errors()
                    if "error" in item.get("ctx", {})
                ),
                "model_declaration.invalid",
            )
        ) from error


def installed_declaration(model_id: str) -> AlphaModelDeclaration:
    """An installed model's declaration, beside its adapter.

    Args:
        model_id: The adapter's id.

    Returns:
        The declaration.
    """
    return read_declaration(_ADAPTERS / f"{model_id}.model.yaml")


__all__ = [
    "AlphaModelDeclaration",
    "ParameterType",
    "installed_declaration",
    "read_declaration",
    "typed_value",
]
