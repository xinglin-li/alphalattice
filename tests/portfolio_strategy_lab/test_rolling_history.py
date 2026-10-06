"""Append-only rolling return heads reuse sealed decision publications."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    advance_decision_state,
)
from alphalattice.investment.portfolio_strategy_lab.publication.portfolio_ledger import (
    PortfolioLedgerStore,
)
from alphalattice.investment.portfolio_strategy_lab.publication.rolling_history import (
    RollingHistoryError,
    RollingReportBaseline,
    RollingReportObservation,
    append_rolling_report_head,
    project_rolling_report,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.synthetic_numerical import (
    build_numerical,
    prepared_for,
    snapshot_for,
)


@pytest.fixture(scope="module")
def rolling_fixture():
    n = build_numerical()
    prefix = []
    previous = None
    # The first three updates close the baseline's last holding period; the
    # next two updates settle the first two post-baseline periods.
    for index in range(n.first, n.first + 7):
        previous = advance_decision_state(
            checkpoint=n.checkpoint,
            previous=previous,
            prepared=prepared_for(n, index) if index < n.first + 5 else None,
            observed=snapshot_for(n, index),
            plan_hash=canonical_hash(index),
            published_at=datetime(2026, 9, 7, tzinfo=UTC),
        )
        prefix.append(previous)
    base_last = prefix[3]
    return n, tuple(prefix), base_last


def _baseline(n, prefix, base_last):
    start = n.sessions[n.first]
    end = n.sessions[n.first + 1]
    return RollingReportBaseline(
        report_hash=canonical_hash("base-report"),
        result_hash=canonical_hash("base-result"),
        program_hash=canonical_hash("base-program"),
        checkpoint_hash=n.checkpoint.history_hash,
        strategy_package_id=n.checkpoint.package.strategy_id,
        strategy_package_hash=n.checkpoint.package.package_hash,
        cost_bps_per_side=5,
        formation_start=start,
        formation_end=end,
        baseline_observed_through=base_last.observed_through,
        base_last_holding_end_session=n.sessions[n.first + 3],
        first_parent_publication_hash=prefix[1].content_hash,
        baseline_claim="BACKTEST_ONLY_NOT_POINT_IN_TIME",
        observations=(
            RollingReportObservation(
                formation_session=start, net_simple_return=0.01, benchmark_simple_return=0.005
            ),
            RollingReportObservation(
                formation_session=end,
                net_simple_return=-0.002,
                benchmark_simple_return=-0.001,
            ),
        ),
    )


def test_two_updates_append_only_head_projects_the_verified_chain(rolling_fixture):
    n, prefix, base_last = rolling_fixture
    baseline = _baseline(n, prefix, base_last)
    publications = prefix[2:7]
    heads = []
    previous = None
    for publication in publications:
        previous = append_rolling_report_head(
            baseline=baseline,
            previous_head=previous,
            publication=publication,
            checkpoint=n.checkpoint,
        )
        heads.append(previous)

    body = project_rolling_report(
        baseline=baseline,
        heads=tuple(heads),
        publications=publications,
        checkpoint=n.checkpoint,
    )
    assert body["formation_range"] == {
        "start": baseline.formation_start.isoformat(),
        "end": publications[-1].events[0].formation_session.isoformat(),
        "base_end": baseline.formation_end.isoformat(),
    }
    assert body["outcomes_observed_through"] == publications[-1].observed_through.isoformat()
    assert body["claim"] == baseline.baseline_claim
    assert body["outcome_count"] == 3
    assert len(body["curve"]) == len(baseline.observations) + 3
    assert body["observation_count"] == len(body["curve"])
    assert len(body["metrics"]) >= 5
    assert body["curve"][-1]["value"] == pytest.approx(
        100.0 * (1.0 + body["metrics"]["cumulative_return"])
    )
    assert body["curve"][0]["benchmark_simple_return"] == 0.005
    assert all(row["benchmark_simple_return"] is None for row in body["curve"][2:])
    assert all(
        "return" not in row and "net_simple_return" not in row for row in body["open_positions"]
    )

    # A later head is constant-sized: it stores one publication ref and the
    # previous-head ref, never either publication's outcome rows.
    serialized = [
        json.dumps(head.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        for head in heads
    ]
    assert heads[0].increment_outcome_count == 0
    assert heads[0].increment_observed_through < baseline.baseline_observed_through
    assert heads[0].observed_through == baseline.baseline_observed_through
    assert abs(len(serialized[2]) - len(serialized[3])) <= 1
    assert b"events" not in serialized[0] and b"observations" not in serialized[0]
    assert heads[1].increment_publication_hash == publications[1].content_hash
    assert heads[1].previous_head_hash == heads[0].head_hash


def test_rolling_head_detects_tampered_earlier_increment(rolling_fixture):
    n, prefix, base_last = rolling_fixture
    baseline = _baseline(n, prefix, base_last)
    publications = prefix[2:7]
    heads = []
    previous = None
    for publication in publications:
        previous = append_rolling_report_head(
            baseline=baseline,
            previous_head=previous,
            publication=publication,
            checkpoint=n.checkpoint,
        )
        heads.append(previous)

    original = publications[1]
    event = original.events[0]
    altered = type(event).create(
        **{
            **{
                name: getattr(event, name)
                for name in type(event).model_fields
                if name != "content_hash"
            },
            "net_return_5bps": event.net_return_5bps + 0.001,
        }
    )
    tampered = type(original).create(
        **{
            **{
                name: getattr(original, name)
                for name in type(original).model_fields
                if name != "content_hash"
            },
            "events": (altered, *original.events[1:]),
        }
    )
    with pytest.raises(RollingHistoryError, match="head_chain_invalid"):
        project_rolling_report(
            baseline=baseline,
            heads=tuple(heads),
            publications=(publications[0], tampered, *publications[2:]),
            checkpoint=n.checkpoint,
        )


def test_rolling_report_refuses_foreign_checkpoint_and_unsupported_cost_lane(rolling_fixture):
    n, prefix, base_last = rolling_fixture
    baseline = _baseline(n, prefix, base_last)
    foreign_baseline = RollingReportBaseline.model_validate(
        {**baseline.model_dump(mode="json"), "checkpoint_hash": canonical_hash("foreign")}
    )
    with pytest.raises(RollingHistoryError, match="foreign_checkpoint_or_package"):
        append_rolling_report_head(
            baseline=foreign_baseline,
            previous_head=None,
            publication=prefix[2],
            checkpoint=n.checkpoint,
        )

    with pytest.raises(ValueError):
        RollingReportBaseline.model_validate(
            {**baseline.model_dump(mode="json"), "cost_bps_per_side": 20}
        )


def test_rolling_head_store_reopens_by_hash_and_detects_tamper(rolling_fixture, tmp_path):
    n, prefix, base_last = rolling_fixture
    baseline = _baseline(n, prefix, base_last)
    head = append_rolling_report_head(
        baseline=baseline,
        previous_head=None,
        publication=prefix[2],
        checkpoint=n.checkpoint,
    )
    store = PortfolioLedgerStore(tmp_path)
    # The owner refuses to publish a head before the exact base and increment
    # references are in this public ledger.
    with pytest.raises(ValueError, match="artifact_missing"):
        store.publish_rolling_report_head(head)
    store.content.publish_model(
        category="rolling-report-heads", value=head, identity_field="head_hash"
    )
    assert store.has_rolling_report_head(head.head_hash)
    assert store.load_rolling_report_head(head.head_hash) == head
    path = store.root / "rolling-report-heads" / f"{head.head_hash}.json"
    original = path.read_bytes()
    tampered = original.replace(b'"increment_outcome_count":0', b'"increment_outcome_count":1')
    assert tampered != original
    path.write_bytes(tampered)
    try:
        with pytest.raises(ValueError, match="artifact_tampered"):
            store.load_rolling_report_head(head.head_hash)
    finally:
        path.write_bytes(original)


class _StoredReportReference(BaseModel):
    report_hash: str
    program_hash: str


class _StoredResultReference(BaseModel):
    result_hash: str
    report_hash: str
    program_hash: str


def _snapshot_files(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_two_daily_heads_append_only_two_refs_each_and_detect_earlier_file_tamper(
    rolling_fixture, tmp_path, monkeypatch
):
    n, prefix, base_last = rolling_fixture
    baseline = _baseline(n, prefix, base_last)
    store = PortfolioLedgerStore(tmp_path)

    report_ref = _StoredReportReference(
        report_hash=canonical_hash({"program_hash": baseline.program_hash}),
        program_hash=baseline.program_hash,
    )
    result_ref = _StoredResultReference(
        result_hash=canonical_hash(
            {"report_hash": report_ref.report_hash, "program_hash": baseline.program_hash}
        ),
        report_hash=report_ref.report_hash,
        program_hash=baseline.program_hash,
    )
    store.content.publish_model(category="reports", value=report_ref, identity_field="report_hash")
    store.content.publish_model(category="results", value=result_ref, identity_field="result_hash")
    # The focused storage test uses tiny references, while exercising the real ledger's
    # publication binding and content-addressed decision/head files.
    monkeypatch.setattr(
        store,
        "load_report",
        lambda report_hash: SimpleNamespace(
            report_hash=report_ref.report_hash, program_hash=baseline.program_hash
        ),
    )
    monkeypatch.setattr(
        store,
        "load_result",
        lambda result_hash: SimpleNamespace(
            report_hash=report_ref.report_hash, program_hash=baseline.program_hash
        ),
    )

    # The first daily increment points to an already sealed predecessor.
    predecessor = prefix[1]
    store.content.publish_model(
        category="decision-updates", value=predecessor, identity_field="content_hash"
    )
    baseline_files = _snapshot_files(store.root)
    report_path = store.root / "reports" / f"{report_ref.report_hash}.json"
    result_path = store.root / "results" / f"{result_ref.result_hash}.json"
    report_bytes = report_path.read_bytes()
    result_bytes = result_path.read_bytes()

    previous_head = None
    previous_publication = predecessor
    snapshots = []
    heads = []
    publications = prefix[2:4]
    for publication in publications:
        assert publication.parent_hash == previous_publication.content_hash
        store.content.publish_model(
            category="decision-updates", value=publication, identity_field="content_hash"
        )
        head = append_rolling_report_head(
            baseline=baseline,
            previous_head=previous_head,
            publication=publication,
            checkpoint=n.checkpoint,
        )
        store.publish_rolling_report_head(head)
        heads.append(head)
        snapshots.append(_snapshot_files(store.root))
        previous_head = head
        previous_publication = publication

    expected_first_append = {
        f"decision-updates/{publications[0].content_hash}.json",
        f"rolling-report-heads/{heads[0].head_hash}.json",
    }
    expected_second_append = {
        f"decision-updates/{publications[1].content_hash}.json",
        f"rolling-report-heads/{heads[1].head_hash}.json",
    }
    assert set(snapshots[0]) - set(baseline_files) == expected_first_append
    assert set(snapshots[1]) - set(snapshots[0]) == expected_second_append
    assert all(snapshots[0][path] == data for path, data in baseline_files.items())
    assert all(snapshots[1][path] == data for path, data in snapshots[0].items())
    assert report_path.read_bytes() == report_bytes
    assert result_path.read_bytes() == result_bytes
    assert len(snapshots[0][f"rolling-report-heads/{heads[0].head_hash}.json"]) < 4096
    assert len(snapshots[1][f"rolling-report-heads/{heads[1].head_hash}.json"]) < 4096
    assert store.load_rolling_report_head(heads[0].head_hash) == heads[0]

    increment_path = store.root / "decision-updates" / f"{publications[0].content_hash}.json"
    original_increment = increment_path.read_bytes()
    increment_path.write_bytes(original_increment[:-1] + bytes([original_increment[-1] ^ 1]))
    try:
        with pytest.raises(ValueError, match="artifact_tampered"):
            store.content.load_model(
                category="decision-updates",
                content_hash=publications[0].content_hash,
                model=type(publications[0]),
                identity_field="content_hash",
            )
        assert store.load_rolling_report_head(heads[0].head_hash) == heads[0]
    finally:
        increment_path.write_bytes(original_increment)
