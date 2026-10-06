"""An agent's model: the declared adapter and its loader (EX).

An extension is three files where G1's owner says: `extensions/<model_id>.py`, the adapter whose
`fit` and `predict` the agent writes; `extensions/<model_id>.model.yaml`, its declaration; and
`tests/alpha_research/extensions/test_<model_id>.py`, its contract test. Everything else an
adapter answers -- its route, recipe, search domain and fit protocol -- is read from the
declaration. The Host loads an extension by name (never by a static import), so its module is
its own closure: a new model is a new capability identity by rule, and nothing else moves. A
workspace installs one only when a person activates it there.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from .contracts import (
    AlphaModelAdapter,
    AlphaModelFitProtocol,
    AlphaModelNumericalBinding,
    AlphaModelRecipeEnvelope,
    AlphaModelSearchDomainEnvelope,
)
from .declaration import AlphaModelDeclaration, read_declaration, typed_value
from .runtime.numerical_environment import SINGLE_THREAD_RUNTIME_CAPABILITY
from .search_axes import SearchSpace

EXTENSION_PACKAGE = "alphalattice.capabilities.alpha_modeling.extensions"


class DeclaredModelAdapter:
    """An adapter whose route, recipe, domain and protocol its declaration states.

    A subclass in `extensions/<model_id>.py` writes `fit` and `predict`; this base reads
    `<model_id>.model.yaml` beside it.
    """

    def __init__(self) -> None:
        """Read the declaration beside the subclass's module.

        Raises:
            ValueError: `model_extension.module_not_the_declaration` when the module's name is
                not the declaration's model.
        """
        path = Path(str(sys.modules[type(self).__module__].__file__))
        self.declaration: AlphaModelDeclaration = read_declaration(
            path.with_name(f"{path.stem}.model.yaml")
        )
        if self.declaration.model_id != path.stem:
            raise ValueError("model_extension.module_not_the_declaration")
        self.adapter_id: str = self.declaration.model_id
        self.recipe_schema_id: str = f"alpha-model.{self.adapter_id}"
        self.search_domain_schema_id: str = f"alpha-model.{self.adapter_id}.search-domain"
        self.content_format_id: str = f"alpha-model.{self.adapter_id}.estimator"

    def describe_numerical_binding(self) -> AlphaModelNumericalBinding:
        """The model's numerical binding: its module owns its numbers.

        Returns:
            The binding.
        """
        return AlphaModelNumericalBinding.create(
            adapter_id=self.adapter_id,
            estimator_content_format_id=self.content_format_id,
            implementation_owners=(type(self).__module__,),
            deterministic_policy={
                "seed": "the recipe's",
                "fit_protocol": self.declaration.fit_protocol,
            },
            # One thread until a canary proves more (W10); the declaration never states it.
            required_runtime_capabilities=(
                "numpy",
                SINGLE_THREAD_RUNTIME_CAPABILITY,
                *self.declaration.libraries,
            ),
        )

    def fit_protocols(self, recipe: AlphaModelRecipeEnvelope) -> tuple[AlphaModelFitProtocol, ...]:
        """The declared protocol.

        Args:
            recipe: Any recipe of this model.

        Returns:
            The declaration's fit protocol.
        """
        del recipe
        return (self.declaration.fit_protocol,)

    def validate_recipe(self, recipe: AlphaModelRecipeEnvelope) -> dict[str, Any]:
        """A recipe of this model: its parameters, each of its declared type.

        Args:
            recipe: The recipe.

        Returns:
            The parameters.

        Raises:
            ValueError: A recipe routed elsewhere, or parameters the declaration does not state.
        """
        if recipe.adapter_id != self.adapter_id or recipe.recipe_schema_id != self.recipe_schema_id:
            raise ValueError("model_extension.recipe_route_invalid")
        declared = self.declaration.parameters
        governed = {axis.name for axis in self.declaration.search.axes}
        values = dict(recipe.parameters)
        if set(values) != set(declared) or any(
            (value is None and name not in governed)
            or (value is not None and not typed_value(value, declared[name]))
            for name, value in values.items()
        ):
            raise ValueError("model_extension.recipe_parameters_invalid")
        return {name: values[name] for name in declared}

    def declared_search_domain(self) -> AlphaModelSearchDomainEnvelope:
        """The search domain the declaration's axes state, as a mandate seals it.

        Returns:
            The domain.
        """
        return AlphaModelSearchDomainEnvelope.create(
            adapter_id=self.adapter_id,
            recipe_schema_id=self.recipe_schema_id,
            search_domain_schema_id=self.search_domain_schema_id,
            constraints={
                "axes": [
                    axis.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for axis in self.declaration.search.axes
                ]
            },
        )

    def validate_search_domain(self, domain: AlphaModelSearchDomainEnvelope) -> SearchSpace:
        """A domain of this model: axes in the declared format.

        Args:
            domain: The domain.

        Returns:
            Its axes.

        Raises:
            ValueError: A domain routed elsewhere or holding no valid axes.
        """
        if (
            domain.adapter_id != self.adapter_id
            or domain.recipe_schema_id != self.recipe_schema_id
            or domain.search_domain_schema_id != self.search_domain_schema_id
        ):
            raise ValueError("model_extension.search_domain_route_invalid")
        try:
            space: SearchSpace = SearchSpace.model_validate(
                {"axes": domain.constraints.get("axes")}
            )
        except ValidationError as error:
            raise ValueError("model_extension.search_domain_invalid") from error
        return space

    def validate_recipe_for_domain(
        self, *, recipe: AlphaModelRecipeEnvelope, domain: AlphaModelSearchDomainEnvelope
    ) -> dict[str, Any]:
        """A recipe inside a domain: every axis the recipe's choices open holds its value.

        Args:
            recipe: The recipe.
            domain: The domain.

        Returns:
            The parameters.

        Raises:
            ValueError: `ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN` for a value off its axis, or a
                value for an axis the recipe's choices close.
        """
        values = self.validate_recipe(recipe)
        space = self.validate_search_domain(domain)
        point = {
            axis.name: values[axis.name] for axis in space.axes if values.get(axis.name) is not None
        }
        for axis in space.axes:
            held = values.get(axis.name)
            if space.holds(axis, point) != (held is not None) or (
                held is not None and not axis.admits(held)
            ):
                raise ValueError("ALPHA_MODEL_RECIPE_OUTSIDE_SEARCH_DOMAIN")
        return values


def extension_adapter(model_id: str) -> AlphaModelAdapter:
    """An extension's adapter, loaded by name.

    Args:
        model_id: The model.

    Returns:
        Its adapter.

    Raises:
        ValueError: `model_extension.not_found:<model_id>` when no extension module names it.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", model_id):
        raise ValueError(f"model_extension.not_found:{model_id}")
    try:
        module = importlib.import_module(f"{EXTENSION_PACKAGE}.{model_id}")
    except ModuleNotFoundError as error:
        raise ValueError(f"model_extension.not_found:{model_id}") from error
    adapter = getattr(module, "ADAPTER", None)
    instance = adapter() if isinstance(adapter, type) else None
    if not isinstance(instance, DeclaredModelAdapter):
        raise ValueError(f"model_extension.not_found:{model_id}")
    return cast(AlphaModelAdapter, instance)


__all__ = [
    "EXTENSION_PACKAGE",
    "DeclaredModelAdapter",
    "extension_adapter",
]
