"""A recipe's identity is its content, never its names (NM1, LAWS.md ID10).

A recipe model lists its labels -- ids, titles, words that select nothing -- in ``LABELS``, and its
identity is the canonical hash of everything else, at every depth: a nested recipe drops its own
labels too. Renaming a label therefore moves no identity, while any change to what the recipe
computes (its features, model and parameters, schedule, book or risk) does.

The walk follows the declared field types, not the values, so a draft built with
``model_construct`` (whose nested parts may still be plain mappings) hashes exactly as the
validated model does.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic import BaseModel

from alphalattice.kernel.shared_kernel.identity import canonical_hash


def recipe_payload(model: BaseModel, *, exclude: frozenset[str] = frozenset()) -> dict[str, object]:
    """The model's JSON content without its labels, at every depth.

    ``exclude`` names the top-level fields left out besides the labels: the model's own seal. A
    draft may hold its values already in their JSON form, so the dump serializes them as given.
    """
    dumped = model.model_dump(mode="json", exclude=set(exclude), warnings=False)
    return cast(dict[str, object], _without_labels(type(model), dumped))


def recipe_identity(model: BaseModel, *, exclude: frozenset[str] = frozenset()) -> str:
    """The canonical hash of the model's content without its labels."""
    return canonical_hash(recipe_payload(model, exclude=exclude))


def recipe_seal_holds(model: BaseModel, seal: str) -> bool:
    """Whether the model's own seal field holds its identity, ``recipe_identity`` (V451).

    A recipe stored under the rule before NM1, its whole content hashed with its names, no longer
    holds: NM2 retired that compatibility path once no root the tree must read needed it.
    """
    return bool(getattr(model, seal) == recipe_identity(model, exclude=frozenset({seal})))


def _without_labels(annotation: object, dumped: object) -> object:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if not isinstance(dumped, dict):
            return dumped
        labels = cast(frozenset[str], getattr(annotation, "LABELS", frozenset()))
        fields = annotation.model_fields
        return {
            key: _without_labels(fields[key].annotation, item) if key in fields else item
            for key, item in dumped.items()
            if key not in labels
        }
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin in (typing.Union, types.UnionType):
        models = [
            argument
            for argument in arguments
            if isinstance(argument, type) and issubclass(argument, BaseModel)
        ]
        # A recipe never nests two models in one union; one model or none is all this walks.
        return _without_labels(models[0], dumped) if len(models) == 1 else dumped
    if isinstance(origin, type) and issubclass(origin, Mapping) and isinstance(dumped, dict):
        value_type = arguments[1] if len(arguments) == 2 else object
        return {key: _without_labels(value_type, item) for key, item in dumped.items()}
    if isinstance(origin, type) and issubclass(origin, Sequence) and isinstance(dumped, list):
        if origin is tuple and not (len(arguments) == 2 and arguments[1] is Ellipsis):
            return [
                _without_labels(argument, item)
                for argument, item in zip(arguments, dumped, strict=False)
            ]
        element = arguments[0] if arguments else object
        return [_without_labels(element, item) for item in dumped]
    return dumped


__all__ = ["recipe_identity", "recipe_payload", "recipe_seal_holds"]
