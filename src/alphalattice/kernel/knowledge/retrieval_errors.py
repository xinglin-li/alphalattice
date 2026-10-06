"""Stable failures for Workspace-owned local knowledge retrieval."""

from __future__ import annotations

from alphalattice.kernel.shared_kernel.domain.errors import AlphaLatticeError


class KnowledgeRetrievalError(AlphaLatticeError):
    """Fail-closed local retrieval error."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        """Construct a typed retrieval failure without exposing source content.

        Args:
            message: Human-readable failure summary.
            code: Stable machine-readable retrieval failure code.
            retryable: Whether the failure is classified as retryable; this grants no execution
                authority.
        """
        super().__init__(
            message,
            category="retrieval",
            code=code,
            retryable=retryable,
        )


__all__ = ["KnowledgeRetrievalError"]
