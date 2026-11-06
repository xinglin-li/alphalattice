"""Model usage records and stage budgets.

A usage record is content-addressed in fact, reads back across a counter
change, cannot hide a model call behind a zero-usage stage, refuses to combine
two models, keeps counters and nothing else; a repair spends what is left of
the stage budget and an exhausted budget refuses before calling a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alphalattice.evidence.alternative_evidence.publication.artifacts import (
    AlternativeEvidenceArtifactStore,
    AlternativeEvidencePublicationError,
)
from alphalattice.evidence.alternative_evidence.publication.usage import (
    ProviderStageUsageRecord,
    record_provider_stage_usage,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _usage(call_count: int = 1, **overrides: Any) -> Any:
    """One stage total, shaped the way the ledger hands it over."""

    from alphalattice.protocols.actor_execution.usage import ProviderStageUsage

    values: dict[str, Any] = {
        "model_id": "fixture-model",
        "call_count": call_count,
        "elapsed_seconds": 1.5,
        "input_tokens": 10,
        "output_tokens": 4,
        "attempts_with_tokens": call_count,
    }
    values.update(overrides)
    return ProviderStageUsage(**values)


def _no_call_usage() -> Any:
    """A stage total that accounts for no model call at all."""

    return _usage(call_count=0, input_tokens=None, output_tokens=None, attempts_with_tokens=0)


def _legacy_usage_record(**overrides: Any) -> dict[str, Any]:
    """A record shaped and hashed the way they were before the counter existed.

    Not a successor with the field blanked out: the key is absent, and the hash
    covers a payload that never had it. The three such records in this
    workspace reproduce their stored hashes only under this rule.
    """

    payload: dict[str, Any] = {
        "kind": "ProviderStageUsageRecord",
        "stage": "ALTERNATIVE_EVIDENCE_ANALYST",
        "subject_hash": "c" * 64,
        "model_id": "deepseek:fixture@0",
        "call_count": 1,
        "elapsed_seconds": 25.719,
        "usage_available": True,
        "input_tokens": 13157,
        "output_tokens": 4206,
    }
    payload.update(overrides)
    return {**payload, "usage_hash": canonical_hash(payload)}


def test_a_usage_record_is_content_addressed_in_fact_not_only_in_name(
    tmp_path: Path,
) -> None:
    """A usage record is content addressed in fact not only in name."""

    store = AlternativeEvidenceArtifactStore(tmp_path / "artifacts")
    record = record_provider_stage_usage(
        store,
        stage="ALTERNATIVE_EVIDENCE_ANALYST",
        subject_hash="a" * 64,
        usage=_usage(),
        model_call_count=1,
    )
    assert record is not None
    reopened = store.load("provider-stage-usage", record.usage_hash, ProviderStageUsageRecord)
    assert reopened == record
    assert reopened.token_coverage == "COMPLETE"

    stored = record.model_dump(mode="json")
    for field, forged in (
        ("input_tokens", 999999),
        ("output_tokens", 999999),
        ("call_count", 7),
        ("elapsed_seconds", 0.001),
        ("model_id", "some-other-model"),
        ("attempts_with_tokens", 0),
    ):
        tampered = {**stored, field: forged}
        with pytest.raises(ValueError):
            ProviderStageUsageRecord.model_validate(tampered)


def test_a_record_written_before_the_coverage_counter_still_reads_back(
    tmp_path: Path,
) -> None:
    """A record written before the coverage counter still reads back."""

    store = AlternativeEvidenceArtifactStore(tmp_path / "artifacts")
    legacy = _legacy_usage_record()
    record = ProviderStageUsageRecord.model_validate(legacy)
    assert record.attempts_with_tokens is None
    assert record.token_coverage is None, "unstated, not UNAVAILABLE"

    # The file as it actually sits on disk: written before the field existed, so
    # its bytes do not mention it. Read through the store's own reader, which
    # re-validates and requires the identity to be the name it is filed under.
    filed = store.root / "provider-stage-usage" / f"{legacy['usage_hash']}.json"
    filed.parent.mkdir(parents=True, exist_ok=True)
    filed.write_bytes(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert "attempts_with_tokens" not in filed.read_text(encoding="utf-8")

    reopened = store.load("provider-stage-usage", record.usage_hash, ProviderStageUsageRecord)
    assert reopened == record
    assert reopened.usage_hash == legacy["usage_hash"], "the identity it was filed under"

    # And it is never migrated in place: re-publishing would serialise the field
    # this record does not have, so the immutable store refuses rather than
    # quietly rewriting an artifact under a hash it no longer matches.
    with pytest.raises(AlternativeEvidencePublicationError, match="artifact_identity_reused"):
        store.publish("provider-stage-usage", record.usage_hash, record)
    assert filed.read_bytes() == json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    ), "the stored bytes are untouched"

    # Changing a field it always carried still invalidates it.
    for field, forged in (
        ("input_tokens", 1),
        ("call_count", 4),
        ("model_id", "another-model"),
        ("elapsed_seconds", 0.5),
    ):
        with pytest.raises(ValueError):
            ProviderStageUsageRecord.model_validate({**legacy, field: forged})

    # And it cannot be passed off as a successor by naming the counter.
    with pytest.raises(ValueError):
        ProviderStageUsageRecord.model_validate({**legacy, "attempts_with_tokens": 1})


def test_a_stage_that_reports_no_usage_cannot_hide_a_model_call(tmp_path: Path) -> None:
    """A stage that reports no usage cannot hide a model call."""

    store = AlternativeEvidenceArtifactStore(tmp_path / "artifacts")

    def write(usage: Any, model_call_count: int) -> Any:
        return record_provider_stage_usage(
            store,
            stage="ALTERNATIVE_EVIDENCE_ANALYST",
            subject_hash="a" * 64,
            usage=usage,
            model_call_count=model_call_count,
        )

    with pytest.raises(ValueError, match="usage_absent_for_model_calls"):
        write(None, 1)
    with pytest.raises(ValueError, match="usage_call_count_mismatch"):
        write(_no_call_usage(), 1)
    with pytest.raises(ValueError, match="usage_call_count_mismatch"):
        write(_usage(call_count=1), 0)
    with pytest.raises(ValueError, match="usage_call_count_mismatch"):
        write(_usage(call_count=2, attempts_with_tokens=2), 1)

    # The Human and already-prepared paths are unchanged: nothing ran, nothing
    # claims to have run, and no zero is invented for them.
    assert write(None, 0) is None
    assert write(_no_call_usage(), 0) is None
    assert not (store.root / "provider-stage-usage").exists(), "nothing was filed"


def test_a_zero_call_stage_that_measured_something_is_refused(tmp_path: Path) -> None:
    """A zero call stage that measured something is refused."""

    from alphalattice.protocols.actor_execution.usage import ProviderStageUsage

    store = AlternativeEvidenceArtifactStore(tmp_path / "artifacts")

    def write(usage: Any, model_call_count: int = 0) -> Any:
        return record_provider_stage_usage(
            store,
            stage="ALTERNATIVE_EVIDENCE_ANALYST",
            subject_hash="a" * 64,
            usage=usage,
            model_call_count=model_call_count,
        )

    # Tokens reported by a stage that claims to have made no call.
    with pytest.raises(ValueError, match="usage_zero_call_measurement_present"):
        write(
            ProviderStageUsage(
                model_id="fixture",
                call_count=0,
                elapsed_seconds=1.0,
                input_tokens=100,
                output_tokens=20,
                attempts_with_tokens=1,
            )
        )

    # Attempts the Provider counted, with no tokens on them.
    with pytest.raises(ValueError, match="usage_zero_call_measurement_present"):
        write(
            ProviderStageUsage(
                model_id="fixture", call_count=0, elapsed_seconds=1.0, attempts_with_tokens=2
            )
        )

    # A zero-call total that measured nothing is still the quiet path.
    assert write(_no_call_usage()) is None
    assert write(None) is None

    # And nothing reached the store on any of those paths.
    assert not (store.root / "provider-stage-usage").exists()


def test_usage_from_two_different_models_refuses_to_combine() -> None:
    """requirement: a stage total that mixed two models would measure nothing."""

    from alphalattice.protocols.actor_execution.usage import ProviderStageUsage

    one = ProviderStageUsage(model_id="model-a", call_count=1, elapsed_seconds=1.0)
    other = ProviderStageUsage(model_id="model-b", call_count=1, elapsed_seconds=1.0)
    with pytest.raises(ValueError, match="usage_model_id_inconsistent"):
        one.combined(other)
