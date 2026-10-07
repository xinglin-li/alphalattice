"""Public local-QA history holders: exact rows, durable roots and refusal boundaries."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alphalattice.control.workspace_runtime.content_store import (
    ContentAddressedStore,
    ContentAddressedStoreError,
    verified_model_read_scope,
)
from alphalattice.foundation.causal_outcomes.execution.artifacts import (
    local_qa_preparation_retention_inventory,
)
from alphalattice.foundation.causal_outcomes.execution.contracts import (
    LocalQAMarketSnapshot,
    LocalQAOutcomePreparationHead,
    LocalQAOutcomePreparationMarker,
    LocalQAOutcomePreparationPart,
    LocalQAOutcomePreparationSourcePrefix,
)
from alphalattice.foundation.causal_outcomes.execution.methods import build_one_session_recipe
from alphalattice.foundation.causal_outcomes.execution.readers import (
    PreparedLocalQASnapshotRows,
    local_qa_prefix,
    local_qa_snapshot_rows,
    planned_local_qa_schedule,
)
from alphalattice.foundation.market_data_ops.sources.contracts import (
    CorporateActionEvent,
    RawDailyBar,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash


def _source(
    *,
    quote="normal",
    actions=(),
    changed_open=None,
    through=date(2026, 9, 11),
    listing_ids=("A", "B"),
):
    schedule = planned_local_qa_schedule(date(2026, 9, 1), through)
    days = tuple(v.formation_session for v in schedule if v.formation_session <= through)
    bars = tuple(
        RawDailyBar(listing, "synthetic", day, 100.0 + i, 102.0 + i, 99.0 + i, 101.0 + i, 1000)
        for i, day in enumerate(days)
        for listing in listing_ids
    )
    affected = (days[2], "A")
    if quote == "missing":
        bars = tuple(v for v in bars if (v.session_date, v.listing_id) != affected)
    elif quote != "normal":
        value = {
            "zero": 0.0,
            "negative_zero": -0.0,
            "nan": float("nan"),
            "infinity": float("inf"),
        }[quote]
        bars = tuple(
            replace(v, open=value) if (v.session_date, v.listing_id) == affected else v
            for v in bars
        )
    if changed_open is not None:
        bars = tuple(
            replace(v, open=changed_open) if (v.session_date, v.listing_id) == affected else v
            for v in bars
        )
    return LocalQAMarketSnapshot.create(
        source_hash="a" * 64,
        through=days[-1],
        ordered_listing_ids=listing_ids,
        schedule=schedule,
        bars=bars,
        actions=actions,
    )


def _days(source):
    return tuple(
        v.formation_session for v in source.schedule if v.formation_session <= source.through
    )


def _encoded(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _files(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _head(root, inventory):
    path = (
        root
        / "data-operations/execution-outcomes/local-qa-preparation/heads"
        / f"{inventory.current_head_hash}.json"
    )
    return LocalQAOutcomePreparationHead.model_validate_json(path.read_bytes())


@pytest.mark.parametrize("reuse_verified", [False, True])
def test_artifact_preparation_keeps_the_source_owners_complete_validation(
    tmp_path, monkeypatch, reuse_verified
):
    source = _source()
    store = ContentAddressedStore(tmp_path / "source", uri_prefix="qa-source")
    store.publish_model(category="market-inputs", value=source, identity_field="content_hash")
    validate = LocalQAMarketSnapshot.model_validate_json
    validations = []

    def counted(cls, payload, **kwargs):
        validations.append(payload)
        return validate(payload, **kwargs)

    monkeypatch.setattr(LocalQAMarketSnapshot, "model_validate_json", classmethod(counted))
    with verified_model_read_scope(reuse_verified=reuse_verified):
        value = store.load_model(
            category="market-inputs",
            content_hash=source.content_hash,
            model=LocalQAMarketSnapshot,
            identity_field="content_hash",
        )
        prepared = PreparedLocalQASnapshotRows.from_artifact(
            source_root=store.root,
            source_category="market-inputs",
            content_hash=source.content_hash,
        )
        assert len(validations) == 1
        expected, expected_bars, expected_axis = local_qa_snapshot_rows(
            value, sessions=_days(value), through=value.through
        )
        actual, bars, axis = prepared.rows(sessions=_days(value), through=value.through)
        assert _encoded(actual) == _encoded(expected) and axis == expected_axis
        assert bars == expected_bars
        bars.clear()
        assert prepared.rows(sessions=_days(value), through=value.through)[1] == expected_bars


@pytest.mark.parametrize("damage", ["content", "address", "delete"])
def test_artifact_preparation_refuses_a_changed_source_after_the_owners_first_load(
    tmp_path, damage
):
    source = _source()
    store = ContentAddressedStore(tmp_path / "source", uri_prefix="qa-source")
    store.publish_model(category="market-inputs", value=source, identity_field="content_hash")
    path = store.root / "market-inputs" / f"{source.content_hash}.json"
    original_stat = path.stat()
    with verified_model_read_scope():
        store.load_model(
            category="market-inputs",
            content_hash=source.content_hash,
            model=LocalQAMarketSnapshot,
            identity_field="content_hash",
        )
        if damage == "delete":
            path.unlink()
        elif damage == "content":
            path.write_bytes(path.read_bytes().replace(b'"synthetic"', b'"tampered"'))
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        else:
            changed = _source(changed_open=123.0)
            path.write_text(changed.model_dump_json(), encoding="utf-8", newline="\n")
        code = "artifact_missing" if damage == "delete" else "artifact_tampered"
        with pytest.raises(ContentAddressedStoreError, match=rf"content_store\.{code}"):
            PreparedLocalQASnapshotRows.from_artifact(
                source_root=store.root,
                source_category="market-inputs",
                content_hash=source.content_hash,
            )


def test_artifact_preparation_still_requires_the_output_capacity_owner(tmp_path):
    source = _source()
    store = ContentAddressedStore(tmp_path / "source", uri_prefix="qa-source")
    store.publish_model(category="market-inputs", value=source, identity_field="content_hash")
    with pytest.raises(ValueError, match=r"causal_outcomes\.qa_preparation_capacity_required"):
        PreparedLocalQASnapshotRows.from_artifact(
            source_root=store.root,
            source_category="market-inputs",
            content_hash=source.content_hash,
            artifact_root=tmp_path / "artifacts",
        )
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("length", [0, 1, 7, 8, 9])
@pytest.mark.parametrize("nulls", ["none", "all", "mixed"])
@pytest.mark.parametrize("chunked", [False, True])
def test_prepared_arrow_buffers_match_the_row_builders_exact_ipc(tmp_path, length, nulls, chunked):
    source = _source(through=date(2026, 9, 24), listing_ids=("A",))
    nan = struct.unpack("<d", bytes.fromhex("a50000000000f87f"))[0]
    special = (-0.0, nan, float("inf"), 0.0, float("-inf"), 0.25)
    bars = tuple(
        replace(bar, open=nan if nulls == "all" else special[i % len(special)])
        if nulls != "none"
        else bar
        for i, bar in enumerate(source.bars)
    )
    source = LocalQAMarketSnapshot.create(
        source_hash=source.source_hash,
        through=source.through,
        ordered_listing_ids=source.ordered_listing_ids,
        schedule=source.schedule,
        bars=bars,
        actions=source.actions,
    )
    # Forward publishes and reloads this captured source before either row builder.
    # Match that public JSON boundary, including its nonfinite-value convention.
    source = LocalQAMarketSnapshot.model_validate_json(
        json.dumps(source.model_dump(mode="json"), separators=(",", ":"))
    )
    days = _days(source)
    sessions = days[:length] if length else (days[-1],)
    expected, _, _ = local_qa_snapshot_rows(source, sessions=sessions, through=source.through)
    assert expected.num_rows == length
    root = tmp_path / "artifacts"
    admissions = []
    cold = PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=admissions.append)
    first, _, _ = cold.rows(sessions=sessions, through=source.through)
    assert _encoded(first) == _encoded(expected)
    head = _head(root, local_qa_preparation_retention_inventory(root))
    assert head.part is not None and head.parts == (head.part,)
    template = (
        pa.concat_tables([expected.slice(0, 1), expected, expected.slice(0, 1)]).combine_chunks()
        if length
        else expected
    )
    count = template.num_rows
    columns = []
    for field, column in zip(template.schema, template.columns, strict=True):
        array = column.combine_chunks()
        if pa.types.is_floating(field.type):
            values = array.to_pylist()
            data = bytearray(array.buffers()[1].to_pybytes())
            width = field.type.bit_width // pa.uint8().bit_width
            for i, value in enumerate(values):
                if value is None:
                    data[i * width : (i + 1) * width] = struct.pack("<d", -123.5)
            buffers = [array.buffers()[0], pa.py_buffer(bytes(data))]
        else:
            buffers = list(array.buffers())
        bits_per_byte = pa.uint8().bit_width
        bitmap = bytearray(bytes([255]) * ((count + bits_per_byte - 1) // bits_per_byte))
        for i in range(count):
            if array.is_null()[i].as_py():
                bitmap[i // bits_per_byte] &= ~(1 << (i % bits_per_byte))
        buffers[0] = pa.py_buffer(bytes(bitmap))
        columns.append(
            pa.Array.from_buffers(field.type, count, buffers, null_count=array.null_count)
        )
    original = pa.Table.from_arrays(columns, schema=template.schema).slice(1, length)
    if chunked:
        split = length // 2
        original = pa.concat_tables([original.slice(0, split), original.slice(split)])
    sink = pa.BufferOutputStream()
    pq.write_table(
        original,
        sink,
        compression="zstd",
        row_group_size=max(1, length // 2) if chunked else max(1, length),
    )
    content = sink.getvalue().to_pybytes()
    part = LocalQAOutcomePreparationPart.create(
        **{
            **head.part.model_dump(mode="python", exclude={"content_hash"}),
            "file_hash": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        },
    )
    replacement = LocalQAOutcomePreparationHead.create(
        **{
            **head.model_dump(mode="python", exclude={"content_hash"}),
            "parts": (part,),
            "part": part,
        },
    )
    marker = LocalQAOutcomePreparationMarker.create(
        current_head_hash=replacement.content_hash, previous_head_hash=None
    )
    preparation = root / "data-operations/execution-outcomes/local-qa-preparation"
    (preparation / "parts" / f"{part.file_hash}.parquet").write_bytes(content)
    for path, value in (
        (preparation / "heads" / f"{replacement.content_hash}.json", replacement),
        (preparation / "markers/current.json", marker),
    ):
        path.write_text(
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
            newline="\n",
        )
    inventory = _files(root)
    writes = len(admissions)
    warm = PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=admissions.append)
    actual, _, _ = warm.rows(sessions=sessions, through=source.through)
    assert _encoded(actual) == _encoded(expected)
    assert actual.schema.equals(template.schema, check_metadata=True)
    assert all(column.num_chunks == 1 for column in actual.columns)
    assert all(column.chunk(0).offset == 0 for column in actual.columns)
    assert _files(root) == inventory and len(admissions) == writes


@pytest.mark.parametrize("damage", ["bar", "action", "duplicate_axis", "unsorted_axis"])
def test_captured_market_snapshot_refuses_outside_or_noncanonical_axes(damage):
    source = _source()
    values = dict(
        source_hash=source.source_hash,
        through=source.through,
        ordered_listing_ids=source.ordered_listing_ids,
        schedule=source.schedule,
        bars=source.bars,
        actions=source.actions,
    )
    if damage == "bar":
        values["bars"] = tuple(
            sorted(
                (replace(source.bars[0], listing_id="OUTSIDE"), *source.bars[1:]),
                key=lambda bar: (bar.session_date, bar.listing_id),
            )
        )
    elif damage == "action":
        values["actions"] = (
            CorporateActionEvent(
                "OUTSIDE", "synthetic", source.through, "CASH_DIVIDEND", cash_amount=0.25
            ),
        )
    elif damage == "duplicate_axis":
        values["ordered_listing_ids"] = ("A", "A", "B")
    else:
        values["ordered_listing_ids"] = ("B", "A")
    with pytest.raises(ValueError, match=r"causal_outcomes\.qa_market_snapshot_invalid"):
        LocalQAMarketSnapshot.create(**values)


@pytest.mark.parametrize("damage", ["bar", "action", "duplicate_axis", "unsorted_axis"])
def test_source_prefix_membership_still_refuses_outside_or_noncanonical_axes(damage):
    source = _source()
    values = dict(
        through=source.through,
        listing_ids=source.ordered_listing_ids,
        recipe_hash=build_one_session_recipe().recipe_hash,
        schedule=tuple(v for v in source.schedule if v.formation_session <= source.through),
        bars=source.bars,
        actions=source.actions,
    )
    if damage == "bar":
        values["bars"] = tuple(
            sorted(
                (replace(source.bars[0], listing_id="OUTSIDE"), *source.bars[1:]),
                key=lambda bar: (bar.session_date, bar.listing_id),
            )
        )
    elif damage == "action":
        values["actions"] = (
            CorporateActionEvent(
                "OUTSIDE", "synthetic", source.through, "CASH_DIVIDEND", cash_amount=0.25
            ),
        )
    elif damage == "duplicate_axis":
        values["listing_ids"] = ("A", "A", "B")
    else:
        values["listing_ids"] = ("B", "A")
    with pytest.raises(ValueError, match=r"causal_outcomes\.qa_preparation_source_prefix_invalid"):
        LocalQAOutcomePreparationSourcePrefix(**values)


@pytest.mark.parametrize("quote", ["normal", "missing", "zero", "negative_zero", "nan", "infinity"])
@pytest.mark.parametrize("preparation", ["snapshot", "artifact"])
def test_durable_qa_rows_append_only_new_points_and_keep_cold_warm_fresh_ipc(
    tmp_path, quote, preparation
):
    root = tmp_path / "artifacts"
    admissions = []
    source = _source(
        quote=quote,
        actions=(
            CorporateActionEvent(
                "A", "synthetic", date(2026, 9, 3), "CASH_DIVIDEND", cash_amount=0.25
            ),
            CorporateActionEvent(
                "B", "synthetic", date(2026, 9, 4), "SPLIT", new_shares_per_old_share=2.0
            ),
        ),
    )
    days = _days(source)
    prior = None
    immutable = {}
    for cutoff in days[3:7]:
        captured = local_qa_prefix(source, through=cutoff)
        captured = LocalQAMarketSnapshot.create(
            source_hash=hashlib.sha256(cutoff.isoformat().encode()).hexdigest(),
            through=captured.through,
            ordered_listing_ids=captured.ordered_listing_ids,
            schedule=captured.schedule,
            bars=captured.bars,
            actions=captured.actions,
        )
        sessions = tuple(day for day in days if day <= cutoff)
        if preparation == "artifact":
            store = ContentAddressedStore(tmp_path / "source", uri_prefix="qa-source")
            store.publish_model(
                category="market-inputs", value=captured, identity_field="content_hash"
            )
            prepared = PreparedLocalQASnapshotRows.from_artifact(
                source_root=store.root,
                source_category="market-inputs",
                content_hash=captured.content_hash,
                artifact_root=root,
                capacity=admissions.append,
            )
        else:
            prepared = PreparedLocalQASnapshotRows(
                captured, artifact_root=root, capacity=admissions.append
            )
        expected, expected_bars, expected_axis = local_qa_snapshot_rows(
            captured, sessions=sessions, through=cutoff
        )
        actual, bars, axis = prepared.rows(sessions=sessions, through=cutoff)
        assert _encoded(actual) == _encoded(expected)
        assert actual.schema == expected.schema and len(actual.columns) == 19
        for name in ("entry_source_row_hash", "holding_end_source_row_hash", "row_hash"):
            assert actual[name].to_pylist() == expected[name].to_pylist()
        assert all(column.num_chunks == 1 for column in actual.columns)
        assert all(
            not buffer.is_mutable
            for column in actual.columns
            for buffer in column.chunk(0).buffers()
            if buffer is not None
        )
        assert axis == expected_axis
        assert tuple((key, canonical_hash(asdict(value))) for key, value in bars.items()) == tuple(
            (key, canonical_hash(asdict(value))) for key, value in expected_bars.items()
        )
        inventory = local_qa_preparation_retention_inventory(root)
        head = _head(root, inventory)
        prefix = LocalQAOutcomePreparationSourcePrefix(
            through=cutoff,
            listing_ids=captured.ordered_listing_ids,
            recipe_hash=build_one_session_recipe().recipe_hash,
            schedule=tuple(v for v in captured.schedule if v.formation_session <= cutoff),
            bars=captured.bars,
            actions=captured.actions,
        )
        payload = prefix.model_dump(mode="json")
        assert head.source_prefix_hash == canonical_hash(payload)
        assert set(payload) == {
            "through",
            "listing_ids",
            "recipe_hash",
            "schedule",
            "bars",
            "actions",
        }
        assert canonical_hash(payload) == canonical_hash(
            {
                "through": cutoff,
                "listing_ids": captured.ordered_listing_ids,
                "recipe_hash": prefix.recipe_hash,
                "schedule": tuple(v.model_dump(mode="json") for v in prefix.schedule),
                "bars": tuple(asdict(v) for v in captured.bars),
                "actions": tuple(asdict(v) for v in captured.actions),
            }
        )
        if quote in ("nan", "infinity"):
            value = next(
                v["open"]
                for v in payload["bars"]
                if v["listing_id"] == "A" and v["session_date"] == days[2].isoformat()
            )
            assert math.isnan(value) if quote == "nan" else math.isinf(value)
        assert head.part is not None
        if prior is None:
            assert head.part.role == "BASE" and head.part.row_count == actual.num_rows
        else:
            assert head.previous_head_hash == prior.content_hash
            assert head.part.role == "INCREMENT"
            assert (
                head.part.point_count == head.matured_point_count - prior.matured_point_count == 1
            )
            assert head.part.row_count == 2
        current = _files(root)
        assert all(current[name] == value for name, value in immutable.items())
        immutable = {name: value for name, value in current.items() if "/markers/" not in name}
        assert all("/heads/" in value.relative_path for value in inventory.candidate_files)
        writes = len(admissions)
        bars.clear()
        again, fresh_bars, fresh_axis = prepared.rows(sessions=sessions, through=cutoff)
        assert _encoded(again) == _encoded(expected) and fresh_bars and fresh_axis == axis
        assert again is not actual and len(admissions) == writes
        assert _files(root) == current
        prior = head

    # Predecessor heads are provenance: retained flat parts are sufficient.
    inventory = local_qa_preparation_retention_inventory(root)
    assert len(inventory.candidate_files) == 2
    for value in inventory.candidate_files:
        (root / value.relative_path).unlink()
    current = _files(root)

    # A new Python process must reuse the sealed rows without publishing anything.
    child = """
import base64, json, sys
from datetime import date
from pathlib import Path
import pyarrow as pa
from alphalattice.foundation.causal_outcomes.execution.contracts import LocalQAMarketSnapshot
from alphalattice.foundation.causal_outcomes.execution.readers import PreparedLocalQASnapshotRows
payload = json.load(sys.stdin)
source = LocalQAMarketSnapshot.model_validate_json(payload['snapshot'])
def no_write(size):
    raise AssertionError('fresh read published preparation bytes')
prepared = PreparedLocalQASnapshotRows(source, artifact_root=Path(sys.argv[1]), capacity=no_write)
table = prepared.rows(sessions=tuple(date.fromisoformat(v) for v in payload['sessions']),
                      through=date.fromisoformat(payload['through']))[0]
sink = pa.BufferOutputStream()
with pa.ipc.new_stream(sink, table.schema) as writer:
    writer.write_table(table)
print(base64.b64encode(sink.getvalue().to_pybytes()).decode('ascii'))
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", child, str(root)],
        input=json.dumps(
            {
                "snapshot": json.dumps(captured.model_dump(mode="json")),
                "sessions": [value.isoformat() for value in sessions],
                "through": cutoff.isoformat(),
            }
        ),
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert base64.b64decode(result.stdout.strip()) == _encoded(expected)
    assert _files(root) == current


@pytest.mark.parametrize("change", ["bar", "action", "schedule", "request"])
def test_durable_qa_history_rebuilds_changed_prefix_even_with_same_source_revision(
    tmp_path, change
):
    root = tmp_path / "artifacts"
    original = _source()
    days = _days(original)
    sessions = days[:6]
    cutoff = sessions[-1]
    first = PreparedLocalQASnapshotRows(original, artifact_root=root, capacity=lambda _size: None)
    first.rows(sessions=sessions, through=cutoff)
    old = local_qa_preparation_retention_inventory(root)
    values = {
        "source_hash": original.source_hash,
        "through": original.through,
        "ordered_listing_ids": original.ordered_listing_ids,
        "schedule": original.schedule,
        "bars": original.bars,
        "actions": original.actions,
    }
    if change == "bar":
        values["bars"] = _source(changed_open=123.0).bars
    elif change == "action":
        values["actions"] = (
            CorporateActionEvent("A", "synthetic", days[2], "CASH_DIVIDEND", cash_amount=0.5),
        )
    elif change == "schedule":
        values["schedule"] = (
            original.schedule[0].model_copy(
                update={
                    "formation_close_at": original.schedule[0].formation_close_at
                    + timedelta(minutes=1)
                }
            ),
            *original.schedule[1:],
        )
    else:
        sessions = sessions[::2]
    changed = LocalQAMarketSnapshot.create(**values)
    assert changed.source_hash == original.source_hash
    prepared = PreparedLocalQASnapshotRows(changed, artifact_root=root, capacity=lambda _size: None)
    actual = prepared.rows(sessions=sessions, through=cutoff)[0]
    expected = local_qa_snapshot_rows(changed, sessions=sessions, through=cutoff)[0]
    assert _encoded(actual) == _encoded(expected)
    inventory = local_qa_preparation_retention_inventory(root)
    head = _head(root, inventory)
    assert head.previous_head_hash is None and head.part.role == "BASE"
    assert inventory.previous_head_hash == old.current_head_hash
    assert inventory.current_head_hash != old.current_head_hash
    old_paths = {value.relative_path for value in old.protected_files}
    assert old_paths <= {value.relative_path for value in inventory.protected_files}


def test_durable_qa_warm_hit_validates_effective_future_action_and_mapping_isolation(tmp_path):
    root = tmp_path / "artifacts"
    source = _source(
        actions=(CorporateActionEvent("A", "synthetic", date(2026, 9, 10), "CAPITAL_GAIN"),)
    )
    prepared = PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=lambda _size: None)
    sessions = (date(2026, 9, 1), date(2026, 9, 2))
    expected = local_qa_snapshot_rows(source, sessions=sessions, through=date(2026, 9, 9))[0]
    table, bars, _ = prepared.rows(sessions=sessions, through=date(2026, 9, 9))
    before = _files(root)
    bars.clear()
    assert _encoded(table) == _encoded(expected)
    for value in (
        prepared,
        PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=lambda _size: None),
    ):
        for cutoff in (date(2026, 9, 10), source.through):
            with pytest.raises(ValueError, match="unsupported corporate action"):
                value.rows(sessions=sessions, through=cutoff)
        result, mapping, _ = value.rows(sessions=sessions, through=date(2026, 9, 9))
        assert _encoded(result) == _encoded(expected) and mapping
        assert _files(root) == before


@pytest.mark.parametrize("damage", ["part", "head", "marker", "missing_part", "missing_head"])
def test_durable_qa_named_artifact_tamper_refuses_without_read_repair(tmp_path, damage):
    root = tmp_path / "artifacts"
    source = _source()
    sessions = _days(source)[:6]
    cutoff = sessions[-1]
    prepared = PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=lambda _size: None)
    prepared.rows(sessions=sessions, through=cutoff)
    inventory = local_qa_preparation_retention_inventory(root)
    category = "parts" if "part" in damage else "heads" if "head" in damage else "markers"
    relative = next(
        value.relative_path
        for value in inventory.protected_files
        if f"/{category}/" in value.relative_path
    )
    target = root / relative
    if damage.startswith("missing"):
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b" ")
    before = _files(root)
    for value in (
        prepared,
        PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=lambda _size: None),
    ):
        with pytest.raises(ValueError, match=r"causal_outcomes\.qa_preparation_"):
            value.rows(sessions=sessions, through=cutoff)
        assert _files(root) == before


def test_qa_retention_inventory_protects_explicit_old_head_and_reports_exact_candidates(tmp_path):
    root = tmp_path / "artifacts"
    heads = []
    for quote in (111.0, 122.0, 133.0):
        source = _source(changed_open=quote)
        sessions = _days(source)[:6]
        PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=lambda _size: None).rows(
            sessions=sessions, through=sessions[-1]
        )
        heads.append(local_qa_preparation_retention_inventory(root).current_head_hash)
    inventory = local_qa_preparation_retention_inventory(root)
    assert inventory.current_head_hash == heads[2] and inventory.previous_head_hash == heads[1]
    assert inventory.candidate_files
    protected = local_qa_preparation_retention_inventory(root, referenced_heads=(heads[0],))
    assert protected.referenced_heads == (heads[0],) and protected.candidate_files == ()
    assert set(_files(root)) == {value.relative_path for value in protected.protected_files}
    for value in (*inventory.protected_files, *inventory.candidate_files):
        content = (root / value.relative_path).read_bytes()
        assert len(content) == value.byte_count
        assert hashlib.sha256(content).hexdigest() == value.file_hash
    with pytest.raises(ValueError, match="qa_preparation_reference_invalid"):
        local_qa_preparation_retention_inventory(root, referenced_heads=("../outside",))
    oldest = next(
        value.relative_path
        for value in protected.protected_files
        if value.relative_path.endswith(f"/{heads[0]}.json")
    )
    (root / oldest).unlink()
    with pytest.raises(ValueError, match="qa_preparation_missing"):
        local_qa_preparation_retention_inventory(root, referenced_heads=(heads[0],))


@pytest.mark.parametrize("changed", [False, True])
def test_unpublished_qa_head_never_grants_authority_and_fresh_retry_is_cold(tmp_path, changed):
    root = tmp_path / "artifacts"
    source = _source()
    sessions = _days(source)[:6]
    cutoff = sessions[-1]
    admissions = []

    def interrupted_capacity(size):
        admissions.append(size)
        if len(admissions) == 3:
            raise ValueError("storage.managed_capacity_exceeded")

    with pytest.raises(ValueError, match=r"storage\.managed_capacity_exceeded"):
        PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=interrupted_capacity).rows(
            sessions=sessions, through=cutoff
        )
    initial = local_qa_preparation_retention_inventory(root)
    assert initial.current_head_hash is None and initial.previous_head_hash is None
    assert initial.protected_files == () and len(initial.candidate_files) == 2
    if changed:
        source = _source(changed_open=123.0)
        # An unpublished head is not read as authority, including corrupt ones.
        orphan = next(
            value for value in initial.candidate_files if "/heads/" in value.relative_path
        )
        (root / orphan.relative_path).write_bytes(b"unpublished interrupted bytes")
    before = _files(root)
    retry_admissions = []
    expected = local_qa_snapshot_rows(source, sessions=sessions, through=cutoff)[0]
    prepared = PreparedLocalQASnapshotRows(
        source, artifact_root=root, capacity=retry_admissions.append
    )
    actual = prepared.rows(sessions=sessions, through=cutoff)[0]
    assert _encoded(actual) == _encoded(expected)
    current = _files(root)
    assert all(current[name] == value for name, value in before.items())
    inventory = local_qa_preparation_retention_inventory(root)
    head = _head(root, inventory)
    assert inventory.previous_head_hash is None and head.previous_head_hash is None
    assert head.part.role == "BASE"
    assert len(retry_admissions) == (3 if changed else 1)
    assert len(inventory.candidate_files) == (2 if changed else 0)
    writes = len(retry_admissions)
    assert _encoded(prepared.rows(sessions=sessions, through=cutoff)[0]) == _encoded(expected)
    assert len(retry_admissions) == writes and _files(root) == current


def test_durable_qa_requires_capacity_and_keeps_marker_last_on_admission_failure(tmp_path):
    root = tmp_path / "artifacts"
    source = _source()
    days = _days(source)
    with pytest.raises(ValueError, match="qa_preparation_capacity_required"):
        PreparedLocalQASnapshotRows(source, artifact_root=root)
    assert not root.exists()
    admissions = []

    def capacity(size):
        admissions.append(size)
        if len(admissions) == 6:
            raise ValueError("storage.managed_capacity_exceeded")

    prepared = PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=capacity)
    prepared.rows(sessions=days[:5], through=days[4])
    original = local_qa_preparation_retention_inventory(root)
    marker = next(value for value in original.protected_files if "/markers/" in value.relative_path)
    with pytest.raises(ValueError, match=r"storage\.managed_capacity_exceeded"):
        prepared.rows(sessions=days[:6], through=days[5])
    assert (
        hashlib.sha256((root / marker.relative_path).read_bytes()).hexdigest() == marker.file_hash
    )
    inventory = local_qa_preparation_retention_inventory(root)
    assert inventory.current_head_hash == original.current_head_hash
    assert len(inventory.candidate_files) == 2
    before = _files(root)
    prepared.rows(sessions=days[:5], through=days[4])
    assert _files(root) == before


def test_historical_qa_publication_keeps_latest_roots_and_next_day_append(tmp_path):
    root = tmp_path / "artifacts"
    source = _source()
    days = _days(source)
    admissions = []
    prepared = PreparedLocalQASnapshotRows(source, artifact_root=root, capacity=admissions.append)

    def rows_at(cutoff):
        sessions = tuple(day for day in days if day <= cutoff)
        expected = local_qa_snapshot_rows(source, sessions=sessions, through=cutoff)[0]
        table = prepared.rows(sessions=sessions, through=cutoff)[0]
        assert _encoded(table) == _encoded(expected)
        return table

    rows_at(date(2026, 9, 9))
    first = local_qa_preparation_retention_inventory(root)
    rows_at(date(2026, 9, 10))
    latest = local_qa_preparation_retention_inventory(root)
    assert latest.previous_head_hash == first.current_head_hash
    before = _files(root)
    writes = len(admissions)

    historical = rows_at(date(2026, 9, 8))
    retained = local_qa_preparation_retention_inventory(root)
    assert retained.current_head_hash == latest.current_head_hash
    assert retained.previous_head_hash == latest.previous_head_hash
    assert retained.protected_files == latest.protected_files
    assert all(_files(root)[name] == value for name, value in before.items())
    assert len(admissions) == writes + 2  # Historical part/head; the marker is unchanged.
    orphan_path = next(
        value.relative_path
        for value in retained.candidate_files
        if "/heads/" in value.relative_path
    )
    historical_head = LocalQAOutcomePreparationHead.model_validate_json(
        (root / orphan_path).read_bytes()
    )
    assert historical_head.through == date(2026, 9, 8)
    assert historical_head.ipc_hash == hashlib.sha256(_encoded(historical)).hexdigest()

    rows_at(date(2026, 9, 11))
    next_day = local_qa_preparation_retention_inventory(root)
    head = _head(root, next_day)
    assert next_day.previous_head_hash == latest.current_head_hash
    assert head.previous_head_hash == latest.current_head_hash
    assert head.part.role == "INCREMENT" and head.part.point_count == 1
    assert head.part.row_count == 2
