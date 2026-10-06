"""Stable failures for governed Live Evidence."""

from __future__ import annotations

from typing import Final

from alphalattice.kernel.shared_kernel.domain.errors import AlphaLatticeError

LIVE_EVIDENCE_FAILURE_CODES: Final = frozenset(
    {
        "live_evidence.source_unknown",
        "live_evidence.source_policy_invalid",
        "live_evidence.source_unavailable",
        "live_evidence.schema_invalid",
        "live_evidence.rights_denied",
        "live_evidence.causal_time_invalid",
        "live_evidence.integrity",
        "live_evidence.publication_failed",
    }
)


class LiveEvidenceError(AlphaLatticeError):
    """Live Evidence exception carrying a bounded stable failure."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        """Create a bounded Live Evidence failure.

        Args:
            message: Safe explanation of the failure.
            code: Installed Live Evidence failure code.
            retryable: Whether source unavailability may be retried.

        Raises:
            ValueError: If the code is unknown or retryability is not allowed.

        """
        if code not in LIVE_EVIDENCE_FAILURE_CODES:
            raise ValueError(f"unknown Live Evidence failure code: {code}")
        if retryable and code != "live_evidence.source_unavailable":
            raise ValueError("only source unavailability can be retryable")
        super().__init__(
            message,
            category="live_evidence",
            code=code,
            retryable=retryable,
        )


__all__ = [
    "LIVE_EVIDENCE_FAILURE_CODES",
    "LiveEvidenceError",
]
