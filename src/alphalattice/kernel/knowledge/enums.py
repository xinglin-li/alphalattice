"""Stable enumeration values for system Knowledge resources."""

from enum import StrEnum


class ResourceNamespace(StrEnum):
    """Identify system, fixture or workspace resource namespaces."""

    SYSTEM = "system"
    FIXTURE = "fixture"
    WORKSPACE = "workspace"


class ResourceKind(StrEnum):
    """Identify guidance, skill, prompt and package-argument resource kinds."""

    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    PROMPT = "prompt"
    PACKAGE_ARG_SPEC = "package_arg_spec"


class ResolutionStatus(StrEnum):
    """Distinguish an exact resolved resource from a typed unsupported outcome."""

    RESOLVED = "resolved"
    UNSUPPORTED = "unsupported"


class ArgumentValueKind(StrEnum):
    """Identify admitted scalar and immutable-list package argument shapes."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    NUMBER = "number"
    STRING = "string"
    STRING_OR_NULL = "string_or_null"
    INTEGER_LIST = "integer_list"
    NUMBER_LIST = "number_list"
    STRING_LIST = "string_list"


__all__ = [
    "ArgumentValueKind",
    "ResolutionStatus",
    "ResourceKind",
    "ResourceNamespace",
]
