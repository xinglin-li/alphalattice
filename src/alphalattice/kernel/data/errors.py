"""Typed failures for deterministic data operations."""

from typing import Final

from alphalattice.kernel.shared_kernel.domain.errors import AlphaLatticeError

DATA_EXECUTION_FAILURE_CODES: Final = frozenset(
    {
        "data.execution_return_invalid_input",
        "data.execution_evidence_conflict",
        "data.execution_return_incomplete",
        "data.execution_return_artifact_invalid",
    }
)


class DataError(AlphaLatticeError):
    """Typed data-domain failure with a stable code and retry semantics."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
        cause_chain: tuple[str, ...] = (),
    ) -> None:
        """Bind a data failure to its code, retry flag, and cause chain."""
        super().__init__(
            message,
            category="data",
            code=code,
            retryable=retryable,
            cause_chain=cause_chain,
        )


class DataProviderError(DataError):
    """Failure while acquiring data from a provider."""

    pass


class DataQualityError(DataError):
    """Failure of deterministic data-quality validation."""

    pass
