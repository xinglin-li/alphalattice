"""Content identity helpers with no domain ownership.

Two successor rules live here, and which one a contract may use is decided by its
*shape* rather than by taste. ``populated_identity`` drops every null member,
recursively, and is the rule for a contract family that nests nothing but itself.
``successor_identity_payload`` drops a named set of successor fields, and only
when none of them is filled; it is the rule for a contract that nests foreign
sealed bindings or carries optional measurements whose absence is itself content.

Neither can replace the other and neither is a general default. They sit together
so the choice is made once, in front of both, instead of being restated per Desk
where it can drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pydantic import BaseModel, TypeAdapter

# ``json.dumps`` with these options builds a new encoder on every call; the
# encoder is stateless, so one instance encodes every value to the same bytes.
_CANONICAL_ENCODER = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=str)

# A schema's prose: a model's and an enum's docstring, a field's description, a title
# (pydantic's own restates a name the structure binds already, a property's key or a
# definition's) and examples. A person or an agent reads them; no identity binds them (SC3).
_SCHEMA_PROSE = frozenset({"title", "description", "examples"})
# The keywords whose value maps names to schemas: the names are structure, each schema is
# walked. A property may itself be called ``title`` or ``description``.
_SCHEMA_MAPS = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)
# The keywords whose value is a schema or a list of them.
_SCHEMA_VALUES = frozenset(
    {
        "items",
        "prefixItems",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "allOf",
        "anyOf",
        "oneOf",
    }
)


def canonical_json(value: object) -> str:
    """The canonical text whose UTF-8 SHA-256 is ``canonical_hash``.

    Exposed for an owner that streams a large payload through the digest in
    pieces; the pieces must concatenate to exactly this text.
    """
    return _CANONICAL_ENCODER.encode(value)


def canonical_hash(value: object) -> str:
    """Hash canonical JSON-safe contract content, never a payload or local path."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def schema_structure(
    schema: type[BaseModel] | TypeAdapter[Any] | Mapping[str, object],
) -> dict[str, object]:
    """Return a JSON schema without its prose: what every schema hash binds (SC3, SH).

    The structure is the types, constraints, required fields, defaults and definitions;
    the prose (docstrings, descriptions, titles, examples) is left out, so a contract's
    words can change while every identity that hashes its schema holds.

    Args:
        schema: A model, a type adapter, or a schema already built from one (a reader's
            choices filled in).

    Returns:
        The schema's structure, ready for ``canonical_hash``.
    """
    if isinstance(schema, Mapping):
        built = schema
    elif isinstance(schema, type):
        built = cast("type[BaseModel]", schema).model_json_schema()
    else:
        built = schema.json_schema()
    return cast(dict[str, object], _schema_structure(built))


def _schema_structure(schema: object) -> object:
    if not isinstance(schema, Mapping):
        return schema  # a boolean schema
    kept: dict[str, object] = {}
    for keyword, value in cast(Mapping[str, object], schema).items():
        if keyword in _SCHEMA_PROSE:
            continue
        if keyword in _SCHEMA_MAPS and isinstance(value, Mapping):
            named = cast(Mapping[str, object], value)
            kept[keyword] = {name: _schema_structure(item) for name, item in named.items()}
        elif keyword in _SCHEMA_VALUES:
            kept[keyword] = (
                [_schema_structure(item) for item in cast(list[object], value)]
                if isinstance(value, list)
                else _schema_structure(value)
            )
        else:
            # Any other keyword's value is data (a type, a bound, a default, a constant, an
            # enumeration, a reference, a discriminator), kept as written.
            kept[keyword] = value
    return kept


def successor_identity_payload(
    payload: dict[str, object], successor_fields: tuple[str, ...]
) -> dict[str, object]:
    """Drop a named set of successor fields when none of them is filled.

    The rule for a contract that nests foreign sealed bindings, or that carries
    genuinely optional *measurements*. A portfolio beta that is ``None`` because
    the exposure was unmeasurable is asserted content rather than an absent
    field, and dropping it would move the identity of every artifact already
    sealed with it -- which is why ``populated_identity`` below cannot own these.

    The absence of the whole named set is the discriminator, and that is what
    makes this safe. There is no version flag for a forger to set, and an
    artifact that omits its successor fields cannot pass itself off as one that
    has them: a verifier reads exactly the same absence and refuses to grant it
    whatever those fields establish.
    """
    if any(payload.get(name) not in (None, [], ()) for name in successor_fields):
        return payload
    return {key: value for key, value in payload.items() if key not in successor_fields}


def populated(value: object) -> object:
    """Drop null members so a field added later cannot move a sealed identity.

    Evidence sealed before a member existed carries no key for it, while a
    contract that has since grown the member reports it as ``None``. Hashing the
    declared shape rather than the asserted content would therefore break
    readback of exactly the artifacts a successor change must preserve.

    The recursion is only sound for a contract family that nests nothing but
    itself. A contract embedding a foreign sealed binding -- an actor submission,
    say -- must keep the plain content hash, because that binding's own identity
    already counted its optional members.
    """
    if isinstance(value, dict):
        return {
            key: populated(item)
            for key, item in cast(Mapping[str, object], value).items()
            if item is not None
        }
    if isinstance(value, list | tuple):
        return [populated(item) for item in cast(list[object], list(value))]
    return value


def populated_identity(payload: Mapping[str, object]) -> str:
    """Content identity over what an artifact actually asserts."""
    return canonical_hash(populated(dict(payload)))


__all__ = [
    "canonical_hash",
    "canonical_json",
    "populated",
    "populated_identity",
    "schema_structure",
    "successor_identity_payload",
]
