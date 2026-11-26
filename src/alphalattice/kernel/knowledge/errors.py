"""Typed failures for system Knowledge resources."""

from __future__ import annotations

from typing import Final

from alphalattice.kernel.shared_kernel.domain.errors import RegisteredCodeError

KNOWLEDGE_FAILURE_CODES: Final = frozenset(
    {
        "knowledge.invalid_reference",
        "knowledge.reference_unsupported",
        "knowledge.catalog_invalid",
        "knowledge.resource_integrity",
        "knowledge.package_version_mismatch",
        "knowledge.argument_unsupported",
        "knowledge.access_denied",
        "knowledge.dependency_unavailable",
    }
)


class KnowledgeError(RegisteredCodeError):
    """Non-retryable Knowledge exception with a stable failure envelope."""

    category = "knowledge"
    label = "Knowledge"
    codes = KNOWLEDGE_FAILURE_CODES


knowledge_failure = KnowledgeError.typed_failure


__all__ = ["KNOWLEDGE_FAILURE_CODES", "KnowledgeError"]
