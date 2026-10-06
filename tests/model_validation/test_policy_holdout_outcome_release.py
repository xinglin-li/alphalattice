"""Claim-bounded execution-outcome release coverage for Portfolio Validation."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.foundation.causal_outcomes.execution.artifacts import (
    _ExecutionOutcomeArtifactStore,
)
from alphalattice.foundation.causal_outcomes.execution.compile import _schema
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    CausalExecutionOutcomeManifest,
    PolicyHoldoutExecutionOutcomeAuthority,
    PolicyHoldoutExecutionOutcomeRelease,
)
from alphalattice.foundation.causal_outcomes.execution.readers import (
    CausalExecutionOutcomeDevelopmentReader,
    CausalExecutionOutcomePolicyHoldoutReader,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash

PLAYPEN_ROOT = Path(__file__).resolve().parents[2]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"


def _authority() -> PolicyHoldoutExecutionOutcomeAuthority:
    formations = tuple(date(2025, 1, 2) + timedelta(days=value) for value in range(252))
    values = {
        "claim_hash": "1" * 64,
        "mandate_hash": "2" * 64,
        "slate_hash": "3" * 64,
        "task_id": "validation-task",
        "source_manifest_ref": "playpen://data-operations/execution-outcomes/manifests/source",
        "source_snapshot_hash": "4" * 64,
        "alpha_program_hashes": ("5" * 64,),
        "validation_numerical_binding_hash": "6" * 64,
        "embargo_session": date(2025, 1, 1),
        "formation_sessions": formations,
        "claimed_at": datetime(2026, 8, 11, tzinfo=UTC),
    }
    return PolicyHoldoutExecutionOutcomeAuthority(
        **values,
        authority_hash=canonical_hash(
            PolicyHoldoutExecutionOutcomeAuthority.model_construct(
                **values,
                authority_hash="",
            ).model_dump(mode="json", exclude={"authority_hash"})
        ),
    )


def _row(session: date) -> dict[str, object]:
    clock = datetime.combine(session, datetime.min.time(), tzinfo=UTC)
    values: dict[str, object] = {
        "listing_id": "listing",
        "symbol": "TEST",
        "sequence": session.toordinal(),
        "formation_session": session,
        "formation_close_at": clock,
        "entry_session": session + timedelta(days=1),
        "entry_open_at": clock + timedelta(days=1),
        "holding_end_session": session + timedelta(days=2),
        "holding_end_open_at": clock + timedelta(days=2),
        "actual_session_span": 2,
        "entry_status": "ASSUMED_ELIGIBLE_FROM_DAILY_BAR",
        "holding_end_status": "ASSUMED_ELIGIBLE_FROM_DAILY_BAR",
        "entry_open_split_adjusted": 100.0,
        "holding_end_open_split_adjusted": 101.0,
        "period_dividend_split_adjusted": 0.0,
        "simple_return": 0.01,
        "entry_source_row_hash": canonical_hash((session, "entry")),
        "holding_end_source_row_hash": canonical_hash((session, "holding")),
    }
    values["row_hash"] = canonical_hash(values)
    return values


def _reader(tmp_path: Path) -> tuple[CausalExecutionOutcomePolicyHoldoutReader, object]:
    artifacts = _ExecutionOutcomeArtifactStore(tmp_path)
    development_session = date(2025, 1, 1)
    authorized_session = date(2025, 1, 2)
    unauthorized_session = date(2025, 1, 3)
    extension_session = date(2025, 1, 4)
    development = artifacts.publish_chunk(
        split="DEVELOPMENT",
        table=pa.Table.from_pylist([_row(development_session)], schema=_schema()),
    )
    sealed = artifacts.publish_chunk(
        split="SEALED_HOLDOUT",
        table=pa.Table.from_pylist(
            [_row(authorized_session), _row(unauthorized_session)], schema=_schema()
        ),
    )
    extension = artifacts.publish_chunk(
        split="SEALED_HOLDOUT",
        table=pa.Table.from_pylist([_row(extension_session)], schema=_schema()),
    )
    values = {
        "research_cadence": "DAILY",
        "market_as_of": date(2025, 1, 5),
        "listing_ids": ("listing",),
        "listing_set_hash": canonical_hash(("listing",)),
        "schedule_hash": "1" * 64,
        "ordered_session_triples_hash": "2" * 64,
        "source_rows_semantic_hash": "3" * 64,
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "development_chunks": (development,),
        "sealed_holdout_chunks": (sealed,),
        "development_formation_count": 1,
        "sealed_holdout_formation_count": 2,
        "limitations": ("test",),
    }
    manifest = CausalExecutionOutcomeManifest(
        **values,
        snapshot_hash=canonical_hash(
            CausalExecutionOutcomeManifest.model_construct(**values, snapshot_hash="").model_dump(
                mode="json", exclude={"snapshot_hash"}
            )
        ),
    )
    descriptor = artifacts.publish_json(
        category="manifests",
        payload=manifest.model_dump(mode="json"),
        identity_field="snapshot_hash",
    )
    reader = object.__new__(CausalExecutionOutcomePolicyHoldoutReader)
    reader.artifacts = artifacts
    reader.manifest_ref = descriptor.uri
    reader.formation_sessions = (authorized_session, extension_session)
    reader._verified_tables = {}
    reader.release = SimpleNamespace(extension_chunk=extension)
    return reader, sealed


def test_policy_holdout_reader_releases_only_the_exact_authorized_axis(tmp_path: Path) -> None:
    reader, _sealed = _reader(tmp_path)
    requested = (date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 4))

    table = reader.read_development_sessions(reader.manifest_ref, requested)

    assert tuple(table["formation_session"].to_pylist()) == requested
    with pytest.raises(ValueError, match="exceeds authority"):
        reader.read_development_sessions(reader.manifest_ref, (date(2025, 1, 3),))
    with pytest.raises(PermissionError, match="GENERIC_SEALED"):
        reader.read_sealed_holdout(reader.manifest_ref)


def test_policy_holdout_reader_rejects_a_fabricated_claim_capability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="authority is unavailable"):
        CausalExecutionOutcomePolicyHoldoutReader(
            market_store=SimpleNamespace(),  # type: ignore[arg-type]
            artifact_root=tmp_path,
            authority=_authority(),
        )


def test_existing_release_index_requires_an_exact_schema(tmp_path: Path) -> None:
    reader, _sealed = _reader(tmp_path)
    claim_hash = "7" * 64
    values = {
        "claim_hash": claim_hash,
        "source_snapshot_hash": "1" * 64,
        "source_listing_set_hash": "2" * 64,
        "universe_manifest_revision": "3" * 64,
        "source_watermark_hash": "4" * 64,
        "embargo_session": date(2025, 1, 1),
        "formation_sessions": tuple(
            date(2025, 1, 4) - timedelta(days=value) for value in reversed(range(252))
        ),
        "extension_chunk": reader.release.extension_chunk,
    }
    release = PolicyHoldoutExecutionOutcomeRelease(
        **values,
        release_hash=canonical_hash(
            PolicyHoldoutExecutionOutcomeRelease.model_construct(
                **values,
                release_hash="",
            ).model_dump(mode="json", exclude={"release_hash"})
        ),
    )
    reader.artifacts.publish_json(
        category="policy-holdout-releases",
        payload=release.model_dump(mode="json"),
        identity_field="release_hash",
    )
    index = reader.artifacts.root / "policy-holdout-releases" / "by-claim" / f"{claim_hash}.json"
    index.parent.mkdir(parents=True)
    payload = {"claim_hash": claim_hash, "release_hash": release.release_hash}
    index.write_text(json.dumps(payload), encoding="utf-8")
    assert (
        CausalExecutionOutcomePolicyHoldoutReader.load_release(
            artifact_root=tmp_path,
            claim_hash=claim_hash,
        )
        == release
    )

    index.write_text(json.dumps({**payload, "unexpected": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="release is unavailable"):
        CausalExecutionOutcomePolicyHoldoutReader.load_release(
            artifact_root=tmp_path,
            claim_hash=claim_hash,
        )


def test_policy_holdout_reader_rejects_value_tamper_with_preserved_metadata(
    tmp_path: Path,
) -> None:
    reader, sealed = _reader(tmp_path)
    path = reader.artifacts.resolve_chunk(sealed)
    table = pq.read_table(path)
    values = table["simple_return"].to_pylist()
    values[0] = 0.5
    changed = table.set_column(
        table.schema.get_field_index("simple_return"),
        "simple_return",
        pa.array(values, type=pa.float64()),
    )
    pq.write_table(changed, path)
    reader._verified_tables.clear()

    with pytest.raises(ValueError, match="chunk is tampered"):
        reader.read_development_sessions(reader.manifest_ref, (date(2025, 1, 2),))


def test_unsealed_snapshot_stays_legacy_readback_and_grants_no_method_authority(
    tmp_path: Path,
) -> None:
    """A snapshot published before the method seam keeps its rows and gains nothing.

    This fixture is exactly that case: a manifest and chunks with no method seal
    beside them. Reading must stay possible -- frozen evidence does not become
    unreadable because a seam was added later -- while every attempt to speak
    for its method fails. Resolving twice must not upgrade it either: authority
    is acquired by publishing, not by looking.
    """

    reader, _sealed = _reader(tmp_path)
    development = CausalExecutionOutcomeDevelopmentReader(tmp_path)
    manifest = CausalExecutionOutcomeManifest.model_validate(
        reader.artifacts.load_json(category="manifests", uri=reader.manifest_ref)
    )

    seal = development.resolve_method_seal(manifest.snapshot_hash)
    assert seal.disposition == "LEGACY_READBACK_ONLY"
    assert seal.binding is None
    with pytest.raises(PermissionError, match="METHOD_AUTHORITY_UNAVAILABLE"):
        _ = seal.method_bound

    rows = reader.read_development_sessions(reader.manifest_ref, (date(2025, 1, 2),))
    assert rows.num_rows == 1
    assert development.resolve_method_seal(manifest.snapshot_hash).disposition == (
        "LEGACY_READBACK_ONLY"
    )


def test_policy_holdout_final_schedule_point_stays_one_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim-bound trailing point is still T -> T+1 -> T+2.

    The offsets moved into the recipe; the Policy Holdout release is frozen at
    the one-session clock and must resolve exactly what it always did.
    """

    reader, _sealed = _reader(tmp_path)
    formation = date(2026, 3, 2)
    sessions = tuple(formation + timedelta(days=value) for value in range(5))
    calendar_rows = [
        {
            "calendar_id": venue,
            "session_date": session,
            "session_open_timestamp": datetime.combine(session, datetime.min.time(), tzinfo=UTC),
            "session_close_timestamp": datetime.combine(session, datetime.min.time(), tzinfo=UTC)
            + timedelta(hours=7),
        }
        for session in sessions
        for venue in ("XNYS", "XNAS")
    ]
    reader.formation_sessions = (formation,)
    monkeypatch.setattr(
        "alphalattice.foundation.causal_outcomes.execution.readers.materialize_calendar_schedule",
        lambda *_args, **_kwargs: pa.Table.from_pylist(calendar_rows),
    )

    point = reader._final_schedule_point(authorized_at=datetime(2026, 3, 2, tzinfo=UTC))

    assert point.formation_session == sessions[0]
    assert point.entry_session == sessions[1]
    assert point.holding_end_session == sessions[2]
    assert point.actual_session_span == 2
    assert point.sequence == 1
