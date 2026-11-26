"""Small construction helper for content-addressed Factor Research contracts."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from alphalattice.kernel.shared_kernel.sealing import seal_model_validated

_Model = TypeVar("_Model", bound=BaseModel)


def seal_contract(  # noqa: UP047
    model_type: type[_Model], identity_field: str, /, **values: object
) -> _Model:
    """Construct and validate one model using its canonical JSON identity."""
    return seal_model_validated(model_type, identity_field, **values)


__all__ = ["seal_contract"]
