"""Sealing a contract record by its content: the hash field holds the canonical hash of the rest.

Every sealer here hashes the record's canonical JSON without its hash field; they differ only in
what the sealed record is validated from (W2): the given values, that JSON, or the draft's Python
dump. The compatible check reads a sealed record back across an optional field added later.
Callers pass pydantic contract models.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, cast

from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _sealed_identity(model: Any, values: Mapping[str, object], field: str) -> dict[str, Any]:
    """The canonical JSON a record's `field` seals: every other field, the draft unvalidated."""
    identity: dict[str, Any] = model.model_construct(**values, **{field: "0" * 64}).model_dump(
        mode="json", exclude={field}
    )
    return identity


def seal_model[M](model: type[M], values: Mapping[str, object], *, field: str) -> M:
    """The record built from `values`, `field` holding the hash of its canonical JSON.

    `model` is a pydantic contract model: it constructs, dumps and validates.

    Args:
        model: The contract model.
        values: Its other fields.
        field: The field that holds the hash.

    Returns:
        The sealed record, validated from `values`.
    """
    constructor: Any = model
    return cast(
        M, constructor(**values, **{field: canonical_hash(_sealed_identity(model, values, field))})
    )


def seal_model_from_dump[M](model: type[M], values: Mapping[str, object], *, field: str) -> M:
    """The record rebuilt from its canonical JSON, `field` holding that JSON's hash.

    Args:
        model: The contract model.
        values: Its other fields.
        field: The field that holds the hash.

    Returns:
        The sealed record, validated from the JSON its hash covers.
    """
    identity = _sealed_identity(model, values, field)
    constructor: Any = model
    return cast(M, constructor(**identity, **{field: canonical_hash(identity)}))


def seal_model_validated[M](model: type[M], field: str, /, **values: object) -> M:
    """The record validated from the draft's Python dump, `field` holding its JSON's hash.

    Args:
        model: The contract model.
        field: The field that holds the hash.
        **values: Its other fields.

    Returns:
        The sealed record.
    """
    contract: Any = model
    draft = contract.model_construct(**values, **{field: "0" * 64})
    identity = draft.model_dump(mode="json", exclude={field})
    payload = draft.model_dump(exclude={field})
    return cast(M, contract.model_validate({**payload, field: canonical_hash(identity)}))


def validate_hash_compatible(
    record: Any, field: str, *, code: str, exclude: Collection[str] = ()
) -> None:
    """Refuse a record whose `field` is neither the hash of its JSON now nor from before.

    A contract changes compatibly by a field added as optional, `None` by default (LAWS.md
    DA6): a record sealed before the field existed hashed its JSON without it, which is the
    JSON without its unset fields, so either hash is accepted.

    Args:
        record: The pydantic contract record.
        field: The field that holds the hash.
        code: The owner's refusal code.
        exclude: Other fields the hash never covered.

    Raises:
        ValueError: `code`, when neither hash matches.
    """
    excluded = {field, *exclude}
    expected = getattr(record, field)
    current = canonical_hash(record.model_dump(mode="json", exclude=excluded))
    historical = canonical_hash(
        record.model_dump(mode="json", exclude=excluded, exclude_unset=True)
    )
    if expected not in {current, historical}:
        raise ValueError(code)


__all__ = [
    "seal_model",
    "seal_model_from_dump",
    "seal_model_validated",
    "validate_hash_compatible",
]
