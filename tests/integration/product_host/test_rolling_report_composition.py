"""Composition checks for joining verified REPORTs to sealed update publications."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from alphalattice.control.product_host.composition import rolling_portfolio_report as composition
from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
    advance_decision_state,
)
from alphalattice.investment.portfolio_strategy_lab.publication.rolling_history import (
    RollingHistoryError,
    RollingReportHead,
)
from alphalattice.kernel.shared_kernel.identity import canonical_hash
from tests.portfolio_strategy_lab.synthetic_numerical import (
    build_numerical,
    prepared_for,
    snapshot_for,
)


@pytest.fixture(scope="module")
def composition_fixture():
    numerical = build_numerical()
    publications = []
    previous = None
    for index in range(numerical.first, numerical.first + 4):
        previous = advance_decision_state(
            checkpoint=numerical.checkpoint,
            previous=previous,
            prepared=prepared_for(numerical, index),
            observed=snapshot_for(numerical, index),
            plan_hash=canonical_hash(("composition", index)),
            published_at=datetime(2026, 9, 7, tzinfo=UTC),
        )
        publications.append(previous)
    return numerical, tuple(publications)


class _ControlledApplication:
    def __init__(self, checkpoint, *, report_end=None, package_hash=None, cost=10):
        self.task_lookup = []
        self.result = SimpleNamespace(
            result_hash=canonical_hash("composition-result"),
            program_hash=canonical_hash("composition-program"),
            execution_ledger_hash=canonical_hash("composition-execution"),
        )
        self.report = SimpleNamespace(
            report_hash=canonical_hash("composition-report"),
            program_hash=self.result.program_hash,
            window_guard=SimpleNamespace(
                selected_start=checkpoint.initial_book.schedule.formation_session
                - timedelta(days=1),
                selected_end=(
                    checkpoint.initial_book.schedule.formation_session
                    if report_end is None
                    else report_end
                ),
            ),
            economic_ledger_hash=canonical_hash("composition-economics"),
        )
        self.program = SimpleNamespace(
            program_hash=self.result.program_hash,
            strategy_package_hash=(
                checkpoint.package.package_hash if package_hash is None else package_hash
            ),
        )
        self.economics = SimpleNamespace(cost_bps_per_side=Decimal(cost))
        self.series = [
            {
                "session": self.report.window_guard.selected_start.isoformat(),
                "net_simple_return": 0.012,
                "benchmark_simple_return": 0.004,
            },
            {
                "session": self.report.window_guard.selected_end.isoformat(),
                "net_simple_return": -0.003,
                "benchmark_simple_return": -0.001,
            },
        ]

        class Pipeline:
            def __init__(inner):
                inner.owner = self

            def find_for_task(inner, task_id):
                inner.owner.task_lookup.append(task_id)
                return SimpleNamespace(result_hash=inner.owner.result.result_hash)

        class Service:
            def __init__(inner, application):
                inner.owner = application

            def open_result(inner, result_hash):
                assert result_hash == inner.owner.result.result_hash
                return inner.owner.result

            def report_of(inner, result):
                assert result is inner.owner.result
                return inner.owner.report

            def path_readback_of(inner, report, *, execution, economics):
                assert report is inner.owner.report
                assert execution is inner.owner.execution
                assert economics is inner.owner.economics
                return {"series": inner.owner.series}

        self.execution = object()
        self.pipeline = Pipeline()

        class Ledger:
            def __init__(inner):
                inner.owner = self
                inner.heads: dict[str, RollingReportHead] = {}
                inner.published: list[str] = []

            def load_program(inner, program_hash):
                assert program_hash == inner.owner.program.program_hash
                return inner.owner.program

            def load_execution(inner, execution_hash):
                assert execution_hash == inner.owner.result.execution_ledger_hash
                return inner.owner.execution

            def load_economics(inner, economics_hash):
                assert economics_hash == inner.owner.report.economic_ledger_hash
                return inner.owner.economics

            def has_rolling_report_head(inner, head_hash):
                return head_hash in inner.heads

            def publish_rolling_report_head(inner, head):
                inner.published.append(head.head_hash)
                inner.heads[head.head_hash] = head
                return head.head_hash

            def load_rolling_report_head(inner, head_hash):
                return inner.heads[head_hash]

        self.ledger = Ledger()
        self.service_type = Service


@pytest.fixture
def controlled_application(monkeypatch, composition_fixture):
    numerical, _ = composition_fixture
    application = _ControlledApplication(numerical.checkpoint)
    monkeypatch.setattr(composition, "LocalPortfolioResearchService", application.service_type)
    return application


def test_baseline_reopens_exact_task_and_preserves_cost_seam_and_benchmark(
    controlled_application, composition_fixture
):
    numerical, publications = composition_fixture
    baseline, rows = composition._baseline(
        controlled_application, numerical.checkpoint, publications[:1]
    )

    assert controlled_application.task_lookup == [numerical.checkpoint.book_task_id]
    assert baseline.cost_bps_per_side == 10
    assert baseline.formation_end == numerical.checkpoint.initial_book.schedule.formation_session
    assert (
        baseline.base_last_holding_end_session
        == numerical.checkpoint.initial_book.schedule.holding_end_session
    )
    assert baseline.baseline_observed_through == baseline.base_last_holding_end_session
    assert baseline.observations[0].benchmark_simple_return == rows[0]["benchmark_simple_return"]
    assert baseline.first_parent_publication_hash == publications[0].parent_hash


@pytest.mark.parametrize("mismatch", ("missing_task", "package", "report_end"))
def test_baseline_refuses_wrong_task_package_or_report_formation_end(
    mismatch, monkeypatch, composition_fixture
):
    numerical, publications = composition_fixture
    application = _ControlledApplication(numerical.checkpoint)
    monkeypatch.setattr(composition, "LocalPortfolioResearchService", application.service_type)
    if mismatch == "missing_task":
        application.pipeline.find_for_task = lambda task_id: None
        expected = "source_book_missing"
    elif mismatch == "package":
        application.program.strategy_package_hash = canonical_hash("foreign-package")
        expected = "source_book_binding_invalid"
    else:
        application.report.window_guard.selected_end = date(2026, 9, 1)
        expected = "source_book_binding_invalid"

    with pytest.raises(RollingHistoryError, match=expected):
        composition._baseline(application, numerical.checkpoint, publications[:1])


def test_publish_missing_heads_is_idempotent_and_reads_never_write(
    controlled_application, composition_fixture
):
    numerical, publications = composition_fixture
    prefix = publications[:2]
    first = composition.publish_rolling_report_heads(
        application=controlled_application,
        checkpoint=numerical.checkpoint,
        publications=prefix,
    )
    second = composition.publish_rolling_report_heads(
        application=controlled_application,
        checkpoint=numerical.checkpoint,
        publications=prefix,
    )

    assert first == second
    assert controlled_application.ledger.published == list(first)
    writes_before_read = len(controlled_application.ledger.published)
    body = composition.read_rolling_report(
        application=controlled_application,
        checkpoint=numerical.checkpoint,
        publications=prefix,
    )
    assert body["head_storage"] == "SEALED"
    assert controlled_application.ledger.published == [*first]
    assert len(controlled_application.ledger.published) == writes_before_read


def test_read_refuses_corrupt_persisted_head_without_repairing_it(
    controlled_application, composition_fixture
):
    numerical, publications = composition_fixture
    prefix = publications[:1]
    hashes = composition.publish_rolling_report_heads(
        application=controlled_application,
        checkpoint=numerical.checkpoint,
        publications=prefix,
    )
    original = controlled_application.ledger.heads[hashes[0]]
    controlled_application.ledger.heads[hashes[0]] = RollingReportHead.create(
        **{
            **original.model_dump(mode="python", exclude={"head_hash"}),
            "baseline_claim": "tampered",
        }
    )
    writes_before_read = len(controlled_application.ledger.published)

    with pytest.raises(RollingHistoryError, match="head_chain_invalid"):
        composition.read_rolling_report(
            application=controlled_application,
            checkpoint=numerical.checkpoint,
            publications=prefix,
        )
    assert len(controlled_application.ledger.published) == writes_before_read
