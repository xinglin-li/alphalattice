"""Typed exceptions that preserve machine-readable failure information."""

from __future__ import annotations

from typing import ClassVar

from alphalattice.kernel.shared_kernel.domain.models import TypedFailure


class AlphaLatticeError(RuntimeError):
    """Base exception carrying a stable failure envelope."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        code: str,
        retryable: bool = False,
        cause_chain: tuple[str, ...] = (),
    ) -> None:
        """Create an exception and its stable machine-readable failure envelope.

        Args:
            message: Human-readable failure detail.
            category: Failure category interpreted by deterministic callers.
            code: Stable machine-readable refusal code.
            retryable: Whether the failure is classified as retryable.
            cause_chain: Ordered lower-level failure descriptions.
        """
        super().__init__(message)
        self.failure = TypedFailure(
            category=category,
            code=code,
            message=message,
            retryable=retryable,
            cause_chain=cause_chain,
        )


class RegisteredCodeError(AlphaLatticeError):
    """A non-retryable failure whose code its category registers.

    A subclass names its category, the label its unknown-code message uses and its registered
    codes; knowledge, quant and validation each keep one.
    """

    category: ClassVar[str]
    label: ClassVar[str]
    codes: ClassVar[frozenset[str]]

    def __init__(self, message: str, *, code: str, cause_chain: tuple[str, ...] = ()) -> None:
        """Bind the failure envelope of a registered code.

        Args:
            message: Human-readable failure detail.
            code: A code the category registers.
            cause_chain: Ordered lower-level failure descriptions.

        Raises:
            ValueError: If the category does not register ``code``.
        """
        if code not in self.codes:
            raise ValueError(f"unknown {self.label} failure code: {code}")
        super().__init__(
            message, category=self.category, code=code, retryable=False, cause_chain=cause_chain
        )

    @classmethod
    def typed_failure(
        cls, message: str, *, code: str, cause: BaseException | None = None
    ) -> TypedFailure:
        """The typed failure of a registered code, its cause named when there is one.

        Args:
            message: Human-readable failure detail.
            code: A code the category registers.
            cause: The exception that caused it, if any.

        Returns:
            The failure envelope.
        """
        cause_chain = () if cause is None else (f"{type(cause).__name__}: {cause}",)
        return cls(message, code=code, cause_chain=cause_chain).failure


class DomainValidationError(AlphaLatticeError):
    """Refuse a domain value that violates its declared contract."""

    def __init__(self, message: str, *, code: str = "domain.invalid") -> None:
        """Bind a validation failure with its stable code.

        Args:
            message: Human-readable failure detail.
            code: Stable refusal code; the class supplies its default when omitted.
        """
        super().__init__(message, category="validation", code=code)


class WorkspaceSecurityError(AlphaLatticeError):
    """Refuse an unsafe workspace path or security boundary."""

    def __init__(self, message: str, *, code: str = "workspace.path_rejected") -> None:
        """Bind a security failure with its stable code.

        Args:
            message: Human-readable failure detail.
            code: Stable refusal code; the class supplies its default when omitted.
        """
        super().__init__(message, category="security", code=code)


class WorkspaceConflictError(AlphaLatticeError):
    """Report a retryable conflict with another workspace owner."""

    def __init__(self, message: str, *, code: str = "workspace.conflict") -> None:
        """Bind a conflict failure with its stable code.

        Args:
            message: Human-readable failure detail.
            code: Stable refusal code; the class supplies its default when omitted.
        """
        super().__init__(message, category="conflict", code=code, retryable=True)


class WorkspaceCorruptionError(AlphaLatticeError):
    """Report an invalid or incomplete durable workspace state."""

    def __init__(self, message: str, *, code: str = "workspace.corruption") -> None:
        """Bind a corruption failure with its stable code.

        Args:
            message: Human-readable failure detail.
            code: Stable refusal code; the class supplies its default when omitted.
        """
        super().__init__(message, category="corruption", code=code)
