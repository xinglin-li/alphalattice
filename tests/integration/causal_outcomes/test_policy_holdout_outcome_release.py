"""Claim-bounded execution-outcome release coverage for Portfolio Validation."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pytest

from alphalattice.capabilities.domain_contracts.identity import canonical_hash
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
from alphalattice.kernel.identity import blake3_digest, blake3_filename
from tests._support.identity_fixtures import fixture_digest

PLAYPEN_ROOT = Path(__file__).resolve().parents[3]
PLAYPEN_SRC = PLAYPEN_ROOT / "src"


@pytest.fixture
def outcome_catalog(tmp_path, monkeypatch):
    from alphalattice.foundation.data_layer import history

    root = tmp_path / "data-root"
    monkeypatch.setenv("ALPHALATTICE_DATA_ROOT", str(root))
    history.open_history("test-policy", "policy fixture", tmp_path / "workspace", root)
    try:
        yield
    finally:
        history.close_history()


def _authority() -> PolicyHoldoutExecutionOutcomeAuthority:
    formations = tuple(date(2025, 1, 2) + timedelta(days=value) for value in range(252))
    values = {
        "claim_hash": fixture_digest("claim"),
        "mandate_hash": fixture_digest("mandate"),
        "slate_hash": fixture_digest("slate"),
        "task_id": "validation-task",
        "source_manifest_ref": "playpen://data-operations/execution-outcomes/manifests/source",
        "source_snapshot_hash": fixture_digest("source-snapshot"),
        "alpha_program_hashes": (fixture_digest("alpha-program"),),
        "validation_numerical_binding_hash": fixture_digest("validation-numerical-binding"),
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
        "schedule_hash": fixture_digest("schedule"),
        "ordered_session_triples_hash": fixture_digest("session-triples"),
        "source_rows_semantic_hash": fixture_digest("source-rows"),
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


def _outcome_read_cases(artifact_root: Path):
    """The same outcome reads for a fixture or an independently copied QA workspace."""
    reader = CausalExecutionOutcomeDevelopmentReader(artifact_root)
    store = reader.artifacts
    cases = []
    for manifest in store.manifests():
        ref = reader.manifest_uri(manifest.snapshot_hash)
        reader.marker_for_snapshot(manifest.snapshot_hash)
        reader.resolve_method_seal(manifest.snapshot_hash)
        for chunk in (*manifest.development_chunks, *manifest.sealed_holdout_chunks):
            cases.append((f"chunk:{chunk.content_hash}", lambda c=chunk: store.read_chunk(c)))
            cases.append(
                (
                    f"projection:{chunk.content_hash}",
                    lambda c=chunk: store.read_chunk(
                        c, columns=["formation_session", "holding_end_session"]
                    ),
                )
            )

            def verified(c=chunk):
                owner = object.__new__(CausalExecutionOutcomePolicyHoldoutReader)
                owner.artifacts, owner._verified_tables = store, {}
                return owner._verified_table(c)

            cases.append((f"policy:{chunk.content_hash}", verified))
        dates = sorted(
            set(
                pa.concat_tables(
                    [
                        store.read_chunk(c, columns=["formation_session"])
                        for c in manifest.development_chunks
                    ]
                )["formation_session"].to_pylist()
            )
        )
        assert dates
        for selection in (
            tuple(dates),
            (dates[0],),
            tuple(sorted({dates[len(dates) // 2], dates[-1]})),
        ):
            cases.append(
                (
                    f"sessions:{len(selection)}:{selection[0]}",
                    lambda r=ref, chosen=selection: reader.read_development_sessions(r, chosen),
                )
            )
        cases.append(("schedule", lambda r=ref: reader.read_development_schedule(r)))
        for cutoff in (dates[0], dates[len(dates) // 2], dates[-1]):
            cases.append(
                (
                    f"matured:{cutoff}",
                    lambda r=ref, c=cutoff: reader.available_development_sessions(
                        r, holding_end_through=c
                    ),
                )
            )
    return cases


def test_policy_holdout_reader_releases_only_the_exact_authorized_axis(
    tmp_path: Path, outcome_catalog
) -> None:
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


def test_existing_release_index_requires_an_exact_schema(
    tmp_path: Path, outcome_catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader, _sealed = _reader(tmp_path)
    claim_hash = fixture_digest("release-index-claim")
    values = {
        "claim_hash": claim_hash,
        "source_snapshot_hash": fixture_digest("release-source-snapshot"),
        "source_listing_set_hash": fixture_digest("source-listings"),
        "universe_manifest_revision": fixture_digest("universe-revision"),
        "source_watermark_hash": fixture_digest("source-watermark"),
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
    index = (
        reader.artifacts.root
        / "policy-holdout-releases/by-claim"
        / f"{blake3_filename(claim_hash)}.json"
    )
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

    # An outcome is catalog-backed and repeat publication keeps one artifact.
    import pyarrow as pa

    from alphalattice.capabilities.data.errors import DataError
    from alphalattice.foundation.data_layer import history

    root = history.data_root()

    day, at = date(2024, 1, 2), datetime(2024, 1, 2, tzinfo=UTC)
    row = {
        "listing_id": "listing",
        "symbol": "X",
        "sequence": 1,
        "formation_session": day,
        "formation_close_at": at,
        "entry_session": day,
        "entry_open_at": at,
        "holding_end_session": day,
        "holding_end_open_at": at,
        "actual_session_span": 2,
        "entry_status": "verified",
        "holding_end_status": "missing",
        "entry_open_split_adjusted": 1.0,
        "holding_end_open_split_adjusted": None,
        "period_dividend_split_adjusted": 0.0,
        "simple_return": None,
        "entry_source_row_hash": "entry",
        "holding_end_source_row_hash": "exit",
        "row_hash": "row",
    }
    row["row_hash"] = canonical_hash(
        {key: value for key, value in row.items() if key != "row_hash"}
    )
    outcomes = _ExecutionOutcomeArtifactStore(tmp_path / "artifacts")
    values = pa.Table.from_pylist([row], schema=_schema())
    chunk = outcomes.publish_chunk(split="DEVELOPMENT", table=values)
    assert outcomes.publish_chunk(split="DEVELOPMENT", table=values) == chunk
    different = values.set_column(
        values.schema.get_field_index("entry_open_split_adjusted"),
        "entry_open_split_adjusted",
        pa.array([2.0], type=pa.float64()),
    )
    with pytest.raises(DataError, match="different values"):
        outcomes.publish_chunk(split="DEVELOPMENT", table=different)
    assert not outcomes._parquet_path(chunk.split, chunk.content_hash).exists()
    assert outcomes.read_chunk(chunk).equals(values)

    # The upgrade runner maps and compares an old file before switching readers.
    from pyarrow import parquet as pq

    from alphalattice.foundation.causal_outcomes.execution.contracts import (
        CausalExecutionOutcomeMarker,
    )
    from alphalattice.foundation.causal_outcomes.execution.readers import (
        execution_outcome_upgrade_family,
    )
    from alphalattice.foundation.data_layer.upgrade import UpgradeRunner

    old_workspace = tmp_path / "old-workspace"
    old = _ExecutionOutcomeArtifactStore(old_workspace / "artifacts")

    def old_chunk(split, base=chunk, data=values):
        metadata = {
            "alphalattice.snapshot_kind": "CausalExecutionOutcomeChunk",
            "alphalattice.chunk_hash": base.content_hash,
            "alphalattice.split": split,
        }
        meta_hash = blake3_digest(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        value = base.model_copy(
            update={
                "split": split,
                "metadata_hash": meta_hash,
                "uri": old._uri(f"{split.casefold().replace('_', '-')}/chunks", base.content_hash),
            }
        )
        path = old._parquet_path(split, base.content_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            data.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()}),
            path,
        )
        return value, path

    development, old_path = old_chunk("DEVELOPMENT")
    sealed, _ = old_chunk("SEALED_HOLDOUT")
    fields = {
        "kind": "CausalExecutionOutcomeSnapshot",
        "research_cadence": "DAILY",
        "market_as_of": day,
        "listing_ids": ("listing",),
        "listing_set_hash": fixture_digest("old-listings"),
        "schedule_hash": fixture_digest("old-schedule"),
        "ordered_session_triples_hash": fixture_digest("old-session-triples"),
        "source_rows_semantic_hash": fixture_digest("old-source-rows"),
        "price_basis": "open_split_adjusted",
        "return_formula_identity": (
            "formation-close-next-common-open-following-common-open-simple-return"
        ),
        "corporate_action_identity": "provider-split-adjusted-open-and-period-dividend",
        "data_validity_class": "CURRENT_UNIVERSE_RESEARCH_ONLY",
        "development_chunks": (development,),
        "sealed_holdout_chunks": (sealed,),
        "development_formation_count": 1,
        "sealed_holdout_formation_count": 1,
        "limitations": (),
    }
    draft = CausalExecutionOutcomeManifest.model_construct(**fields, snapshot_hash="")
    manifest = CausalExecutionOutcomeManifest(
        **fields,
        snapshot_hash=canonical_hash(draft.model_dump(mode="json", exclude={"snapshot_hash"})),
    )
    ref = old.publish_json(
        category="manifests",
        payload=manifest.model_dump(mode="json"),
        identity_field="snapshot_hash",
    ).uri
    marker_fields = {
        "kind": "CausalExecutionOutcomeMarker",
        "snapshot_hash": manifest.snapshot_hash,
        "manifest_ref": ref,
        "schedule_hash": manifest.schedule_hash,
        "listing_set_hash": manifest.listing_set_hash,
    }
    marker = CausalExecutionOutcomeMarker(
        **marker_fields, marker_hash=canonical_hash(marker_fields)
    )
    old.publish_json(
        category="markers",
        payload=marker.model_dump(mode="json"),
        identity_field="marker_hash",
    )
    # A committed policy release has a separate extension outside its manifest.
    extension_row = row | {"listing_id": "extension"}
    extension_row["row_hash"] = canonical_hash(
        {key: value for key, value in extension_row.items() if key != "row_hash"}
    )
    extension_values = pa.Table.from_pylist([extension_row], schema=_schema())
    extension_hash = canonical_hash(
        {
            "schema": "desktop-causal-execution-outcome",
            "rows": (("extension", str(day), extension_row["row_hash"]),),
        }
    )
    extension_base = chunk.model_copy(update={"content_hash": extension_hash})
    extension, _ = old_chunk("SEALED_HOLDOUT", extension_base, extension_values)
    release_fields = {
        "claim_hash": fixture_digest("claim"),
        "source_snapshot_hash": manifest.snapshot_hash,
        "source_listing_set_hash": manifest.listing_set_hash,
        "universe_manifest_revision": fixture_digest("old-universe-revision"),
        "source_watermark_hash": fixture_digest("old-source-watermark"),
        "embargo_session": day - timedelta(days=253),
        "formation_sessions": tuple(day - timedelta(days=i) for i in reversed(range(252))),
        "extension_chunk": extension,
    }
    draft_release = PolicyHoldoutExecutionOutcomeRelease.model_construct(
        **release_fields, release_hash=""
    )
    release = PolicyHoldoutExecutionOutcomeRelease(
        **release_fields,
        release_hash=canonical_hash(
            draft_release.model_dump(mode="json", exclude={"release_hash"})
        ),
    )
    old.publish_json(
        category="policy-holdout-releases",
        payload=release.model_dump(mode="json"),
        identity_field="release_hash",
    )
    index = (
        old.root
        / "policy-holdout-releases/by-claim"
        / f"{blake3_filename(release.claim_hash)}.json"
    )
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps({"claim_hash": release.claim_hash, "release_hash": release.release_hash})
    )
    cases = _outcome_read_cases(old_workspace / "artifacts")
    before = [(name, read()) for name, read in cases]
    family = execution_outcome_upgrade_family()
    upgrade = UpgradeRunner(
        root,
        history.holding(),
        (old_workspace,),
        (family,),
        lambda *_args: {"fixture": "verified"},
    )
    assert upgrade.run() is True
    artifact = upgrade.id_map[("execution-outcomes", chunk.content_hash)]
    assert uuid.UUID(artifact).version == 7
    assert history.execution_outcome_artifact_id(chunk.content_hash) == artifact
    assert [
        tuple(row)
        for row in history.holding().read(
            "SELECT artifact_id, content_digest FROM artifact WHERE artifact_id = ?", (artifact,)
        )
    ] == [(artifact, chunk.content_hash)]
    assert not history.holding().read(
        "SELECT 1 FROM definition WHERE family = 'provider' AND name = ?",
        ("alphalattice-causal-execution",),
    )
    assert old_path.is_file() and old.read_chunk(chunk).equals(values)
    assert old.read_chunk(extension).equals(extension_values)
    assert uuid.UUID(upgrade.id_map[("execution-outcomes", extension_hash)]).version == 7
    after = [(name, read()) for name, read in cases]
    for (name, first), (other, second) in zip(before, after, strict=True):
        assert name == other
        assert first.equals(second) if isinstance(first, pa.Table) else first == second


def test_policy_holdout_reader_rejects_value_tamper_with_preserved_metadata(
    tmp_path: Path,
    outcome_catalog,
) -> None:
    reader, sealed = _reader(tmp_path)
    from alphalattice.foundation.data_layer import history

    artifact = history.holding().read(
        "SELECT artifact_id FROM artifact WHERE content_digest = ?", (sealed.content_hash,)
    )[0][0]
    history.store().lake.execute(
        "UPDATE lake.results.execution_outcome SET simple_return = 0.5 "
        "WHERE artifact_id = ? AND formation_session = ?",
        (artifact, date(2025, 1, 2)),
    )
    reader._verified_tables.clear()

    with pytest.raises(ValueError, match="chunk is tampered"):
        reader.read_development_sessions(reader.manifest_ref, (date(2025, 1, 2),))


def test_unsealed_snapshot_stays_legacy_readback_and_grants_no_method_authority(
    tmp_path: Path,
    outcome_catalog,
) -> None:
    """Unsealed snapshot stays legacy readback and grants no method authority."""

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome_catalog
) -> None:
    """Policy holdout final schedule point stays one session."""

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
