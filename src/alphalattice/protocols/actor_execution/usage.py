"""What a model stage cost, as counters only.

A protocol, not a framework adapter. It sits beside the actor-execution binding
both Evidence and Oversight already read, and imports nothing from the Agent
framework: the Host loads this to report what a run cost, and it answers
projections and refusals without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TokenCoverage = Literal["COMPLETE", "PARTIAL", "UNAVAILABLE"]
"""How much of a stage the Provider actually reported tokens for."""


@dataclass(frozen=True, slots=True)
class ProviderStageUsage:
    """What one actor stage cost, as the Provider reported it.

    Token counts are optional on purpose. A Provider that reports nothing must
    read as unavailable, never as zero: zero is a measurement and absence is
    not, and the two are easy to confuse once they reach a report.

    `attempts_with_tokens` is carried separately from `call_count` so a stage
    where only some attempts were counted reads as partial rather than as a
    complete total that happens to be small.
    """

    model_id: str
    call_count: int
    elapsed_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts_with_tokens: int = 0

    @property
    def usage_available(self) -> bool:
        """Return whether the Provider reported either token counter."""
        return self.input_tokens is not None or self.output_tokens is not None

    @property
    def token_coverage(self) -> TokenCoverage:
        """Whether every attempt in this stage was counted, some, or none."""
        if self.attempts_with_tokens == 0:
            return "UNAVAILABLE"
        if self.attempts_with_tokens >= self.call_count:
            return "COMPLETE"
        return "PARTIAL"

    def combined(self, other: ProviderStageUsage) -> ProviderStageUsage:
        """Fold one attempt into the stage it belongs to.

        Refuses two different models. A stage total that silently averaged one
        model's tokens with another's would be a measurement of nothing.
        """
        if self.model_id and other.model_id and self.model_id != other.model_id:
            raise ValueError("actor_execution.usage_model_id_inconsistent")

        def add(left: int | None, right: int | None) -> int | None:
            if left is None and right is None:
                return None
            return (left or 0) + (right or 0)

        return ProviderStageUsage(
            model_id=self.model_id or other.model_id,
            call_count=self.call_count + other.call_count,
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
            input_tokens=add(self.input_tokens, other.input_tokens),
            output_tokens=add(self.output_tokens, other.output_tokens),
            attempts_with_tokens=self.attempts_with_tokens + other.attempts_with_tokens,
        )


__all__ = ["ProviderStageUsage", "TokenCoverage"]
