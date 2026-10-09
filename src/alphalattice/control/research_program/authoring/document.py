"""Safe authoring-document parsing for the research-authoring route.

A YAML document selects installed capabilities and parameters. It must never be
able to name a Python module, a callable, or an import path, and it must never
construct a Python object through a YAML tag. `yaml.safe_load` refuses the tags;
this module additionally refuses the key names and shapes that would reintroduce
dynamic import by convention.

The wider rule -- what a committed *request* may say at all -- lives in
``protocols/research_authoring/selection``, because it is a statement about the
shape of a document rather than a behaviour of this runtime. It is re-exported
here for the callers that already reach for it on this path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import yaml  # type: ignore[import-untyped]

from alphalattice.protocols.research_authoring.contracts import AuthoringError
from alphalattice.protocols.research_authoring.selection import (
    FORBIDDEN_KEYS,
    MAXIMUM_DEPTH,
    UnparsableDocument,
    load_safe_yaml_document,
    load_selection_only_document,
    looks_like_python_path,
    require_selection_only_document,
)

_MAXIMUM_DEPTH = MAXIMUM_DEPTH


def load_authoring_document(text: str) -> Mapping[str, Any]:
    """Parse one authoring document, failing closed on dynamic-import shapes."""
    try:
        parsed = load_safe_yaml_document(text)
    except yaml.YAMLError as error:
        raise UnparsableDocument(error) from error
    return require_authoring_document(parsed)


def require_authoring_document(document: object) -> dict[str, Any]:
    """Check an authoring document given as an object, as a parsed YAML one is checked.

    A document sent as an object skipped the YAML entry's guard, so `risk.module`
    was refused as YAML and accepted as an object; both entries run this one
    check.

    Args:
        document: The authored document, parsed or sent as an object.

    Returns:
        The document, unchanged.

    Raises:
        AuthoringError: It is no mapping, or it names a module, a callable or an
            import path.
    """
    if not isinstance(document, dict):
        raise AuthoringError("research_authoring.document_not_a_mapping")
    _reject_dynamic_import(document, depth=0)
    return document


def _reject_dynamic_import(node: object, *, depth: int) -> None:
    if depth > _MAXIMUM_DEPTH:
        raise AuthoringError("research_authoring.document_too_deep")
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(key, str):
                raise AuthoringError("research_authoring.document_key_not_a_string")
            if key.lower() in FORBIDDEN_KEYS:
                raise AuthoringError("research_authoring.document_declares_python_path")
            _reject_dynamic_import(value, depth=depth + 1)
        return
    if isinstance(node, list | tuple):
        for value in node:
            _reject_dynamic_import(value, depth=depth + 1)
        return
    if isinstance(node, str) and looks_like_python_path(node):
        raise AuthoringError("research_authoring.document_declares_python_path")


__all__ = [
    "FORBIDDEN_KEYS",
    "load_authoring_document",
    "load_selection_only_document",
    "require_authoring_document",
    "require_selection_only_document",
]
