"""What each model stage cost, recorded beside the work it paid for.

An append-only family in the store that already holds every evidence artifact.
Counters and elapsed time only: no prompt, no message content and no Provider
payload. A Provider that reports no tokens is recorded as unavailable, because
zero is a measurement and absence is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from alphalattice.evidence.alternative_evidence.contracts import (
    AlternativeEvidenceContract,
    seal_contract,
    validate_contract_identity,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from alphalattice.protocols.actor_execution.usage import TokenCoverage

if TYPE_CHECKING:
    from alphalattice.evidence.alternative_evidence.publication.artifacts import (
        AlternativeEvidenceArtifactStore,
    )
    from alphalattice.protocols.actor_execution.usage import ProviderStageUsage

_HASH = r"^[0-9a-f]{64}$"


class ProviderStageUsageRecord(AlternativeEvidenceContract):
    """One model stage, as the Provider reported it.

    `usage_available` is carried explicitly so a reader never has to decide
    whether a zero is a measurement. When the Provider reported nothing the
    token fields are absent and this is false.
    """

    kind: Literal["ProviderStageUsageRecord"] = "ProviderStageUsageRecord"
    stage: Literal["ALTERNATIVE_EVIDENCE_ANALYST", "CHIEF_RISK_OFFICER_REVIEWER"]
    subject_hash: str = Field(pattern=_HASH)
    model_id: str = Field(min_length=1, max_length=160)
    call_count: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    usage_available: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    attempts_with_tokens: int | None = Field(default=None, ge=0)
    """How many of `call_count` attempts the Provider actually counted.

    Optional because records written before this counter existed cannot say,
    and reporting them as zero would claim a measurement nobody made.
    """

    usage_hash: str = Field(pattern=_HASH)

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_usage(self) -> Self:
        """Validate the recorded usage values and sealed identity."""
        if self.usage_available == (self.input_tokens is None and self.output_tokens is None):
            raise ValueError("alternative_evidence.usage_availability_invalid")
        if self.attempts_with_tokens is not None:
            if self.attempts_with_tokens > self.call_count:
                raise ValueError("alternative_evidence.usage_attempt_count_invalid")
            if self.usage_available != (self.attempts_with_tokens > 0):
                raise ValueError("alternative_evidence.usage_availability_invalid")
        self._validate_identity()
        return self

    def _validate_identity(self) -> None:
        """Recompute `usage_hash` over the payload this record was sealed from.

        Two payloads, one rule each, because `attempts_with_tokens` arrived after
        some records were written and their hash covers a payload that has no
        such key -- not a key set to null, which hashes differently. A record
        that states the counter is therefore verified against the whole payload,
        and one that does not against the payload without it. Neither shape is
        checked under the other's rule, so a legacy record cannot be passed off
        as a successor, and changing any field either of them already carried
        still invalidates the hash it was filed under.
        """
        if self.attempts_with_tokens is not None:
            validate_contract_identity(self, "usage_hash")
            return
        legacy = self.model_dump(mode="json", exclude={"usage_hash", "attempts_with_tokens"})
        if self.usage_hash != canonical_hash(legacy):
            raise ValueError("alternative_evidence.usage_identity_invalid")

    @property
    def token_coverage(self) -> TokenCoverage | None:
        """Whether every attempt was counted, some were, or none -- or unstated."""
        if self.attempts_with_tokens is None:
            return None
        if self.attempts_with_tokens == 0:
            return "UNAVAILABLE"
        if self.attempts_with_tokens >= self.call_count:
            return "COMPLETE"
        return "PARTIAL"


def record_provider_stage_usage(
    artifacts: AlternativeEvidenceArtifactStore,
    *,
    stage: str,
    subject_hash: str,
    usage: ProviderStageUsage | None,
    model_call_count: int,
) -> ProviderStageUsageRecord | None:
    """Publish one usage record, or nothing when no model was involved.

    A stage answered by a Human or an already-prepared submission has no
    Provider usage at all, and inventing a zero for it would be a false
    measurement rather than a missing one.

    `model_call_count` is what the sealed receipt says the stage cost, and it is
    reconciled here rather than trusted: the two were counted at different
    moments once already, and a usage total that quietly disagreed with the
    receipt would be the same defect wearing a different number.
    """
    observed = 0 if usage is None else usage.call_count
    if model_call_count > 0 and usage is None:
        # A receipt that says a model ran, beside a stage that reported nothing,
        # is the defect this reconciliation exists to catch. Returning early
        # here is what let it through.
        raise ValueError(f"alternative_evidence.usage_absent_for_model_calls:{model_call_count}")
    if observed != model_call_count:
        raise ValueError(
            f"alternative_evidence.usage_call_count_mismatch:{observed}!={model_call_count}"
        )
    if usage is not None and usage.call_count == 0:
        # The counts agree at zero, so what is left to check is whether the
        # total agrees with itself. A stage that reports tokens, or attempts the
        # Provider counted, did not run zero calls, and discarding that on the
        # strength of the count alone would drop a real measurement to keep a
        # receipt tidy.
        measured = (
            usage.attempts_with_tokens != 0
            or usage.usage_available
            or usage.input_tokens is not None
            or usage.output_tokens is not None
        )
        if measured:
            raise ValueError(
                "alternative_evidence.usage_zero_call_measurement_present:"
                f"attempts={usage.attempts_with_tokens},"
                f"input={usage.input_tokens},output={usage.output_tokens}"
            )
    if usage is None or usage.call_count == 0:
        # Nothing ran and nothing claims to have run: a Human or an
        # already-prepared submission. There is no usage to file, and inventing
        # a zero for it would be a false measurement rather than a missing one.
        return None
    record = seal_contract(
        ProviderStageUsageRecord,
        "usage_hash",
        stage=stage,
        subject_hash=subject_hash,
        model_id=usage.model_id,
        call_count=usage.call_count,
        elapsed_seconds=round(usage.elapsed_seconds, 3),
        usage_available=usage.usage_available,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        attempts_with_tokens=usage.attempts_with_tokens,
    )
    artifacts.publish("provider-stage-usage", record.usage_hash, record)
    return record


__all__ = ["ProviderStageUsageRecord", "record_provider_stage_usage"]
