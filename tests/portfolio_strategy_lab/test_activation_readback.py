"""The exact book's activation standing through the product's real readers."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlsplit
from uuid import UUID, uuid4

import numpy as np
import pytest

from alphalattice.capabilities.portfolio_backtesting.metrics import evaluate_net_simple_return_path
from alphalattice.interface.local_application.cli import main
from tests.portfolio_strategy_lab.activation_review_support import (
    PACKAGE,
    activation_cli,
    publish_activation_review,
)
from tests.portfolio_strategy_lab.cli_support import _cli
from tests.portfolio_strategy_lab.local_web_support import _json


@pytest.mark.parametrize("reviewed", [False, True], ids=["unreviewed", "reviewed"])
def test_every_activation_surface_reads_the_exact_books_published_review(
    tmp_path, capsys, reviewed, monkeypatch
):
    """Activation surfaces read the exact book's published review without starting research work."""

    from alphalattice.interface.local_application.client import LocalResearchClient
    from tests.alternative_evidence_desk.review_http_support import build_authority
    from tests.portfolio_strategy_lab.activation_review_support import activation_review_host

    with activation_review_host(
        tmp_path / "workspace",
        review_authority_for_report=lambda report: build_authority(
            tmp_path=tmp_path, report=report
        ),
    ) as book:
        live, task_id, result_hash = book.live, str(book.task_id), book.result_hash
        view = publish_activation_review(live, result_hash) if reviewed else None
        # An unrelated current analysis selection must not hide this book's published review.
        live.review.selected_analysis_publication_hash = "f" * 64
        before = (
            len(live.session.task_control_registry.tasks()),
            live.review.artifacts.write_count,
        )
        raw = activation_cli(live, capsys, "result", "show", result_hash)
        standing = raw["review_standing"]
        assert raw["standing"]["activation"] == "A_PERSON_MAY_ACTIVATE"
        # Saved-window facts remain beside the book's review in the combined answer.
        assert raw["selected_window_metrics"]["cumulative_return"] == (
            book.report.window_cumulative_net_wealth - 1.0
        )
        assert raw["selected_window_metrics"]["cost_bps"] == float(
            raw["readouts"]["platform_one_way_cost_bps"]
        )
        assert "annualized_return" in raw["selected_window_metrics"]
        assert raw["selected_window_metric_provenance"]["status"] == (
            "DERIVED_FROM_SEALED_NET_RETURN_PATH"
        )
        assert standing["book_selector"] == {"result_hash": result_hash}
        if view is None:
            assert standing["status"] == "NOT_REVIEWED"
            assert standing["detail"] == "This book has not been reviewed."
            assert standing["coverage"] is standing["cro"] is standing["published_at"] is None
        else:
            assert standing["status"] == "REVIEWED"
            assert standing["review_publication_hash"] == view.publication.publication_hash
            assert standing["published_at"] == view.publication.published_at.isoformat()
            assert standing["evidence_as_of"] == view.dossier.evidence_as_of.isoformat()
            coverage = standing["coverage"]
            assert coverage["prepared"]["scope"] == "ANALYSIS_LINKED_PACKETS"
            assert coverage["prepared"]["with_preparation_receipt"] > 0
            assert coverage["prepared"]["preparation_unknown"] == 0
            assert coverage["analyzed"]["scope_issuers"] == len(view.dossier.issuers)
            assert coverage["reviewed"]["reviewed_ending_weight_coverage"] == (
                view.dossier.coverage.reviewed_ending_weight_coverage
            )
            assert standing["cro"] == {
                "route": view.recommendation.route.value,
                "review_state": view.recommendation.review_state.value,
                "reasons": list(view.recommendation.reasons),
            }
        compact = activation_cli(live, capsys, "result", "show", result_hash, view="compact")
        assert compact["standing"]["statements"] == raw["standing"]["statements"]
        concise = compact["review_standing"]
        # Compact CLI references are executable short names; the standing facts stay whole.
        for field in standing.keys() - {
            "book_selector",
            "review_publication_hash",
            "next_requests",
        }:
            assert concise[field] == standing[field]
        assert result_hash.startswith(concise["book_selector"]["result_hash"])
        if reviewed:
            assert standing["review_publication_hash"].startswith(
                concise["review_publication_hash"]
            )
        positioned = activation_cli(
            live, capsys, "result", "show", result_hash, "--session", book.last_session.isoformat()
        )
        assert positioned["review_standing"] == standing
        assert positioned["selected_window_metrics"] == raw["selected_window_metrics"]
        body = positioned
        store = live.service.application.ledger
        execution = store.load_execution(book.report.execution_ledger_hash)
        economics = store.load_economics(book.report.economic_ledger_hash)
        net = np.asarray(
            [
                economics.net_simple_returns[i]
                for i, session in enumerate(execution.formation_sessions)
                if book.report.window_guard.selected_start
                <= session
                <= book.report.window_guard.selected_end
            ],
            dtype=np.float64,
        )
        expected = evaluate_net_simple_return_path(net_simple_returns=net)
        absences = body["selected_window_metric_absences"]
        benchmark_absences = {
            "information_ratio",
            "benchmark_relative_return",
            "beta",
            "tracking_error",
            "zero_cash_jensen_alpha",
        }
        assert set(absences) == benchmark_absences | (
            {"sortino"} if not np.isfinite(expected.sortino) else set()
        )
        assert all(
            set(absences[name]) == {"status", "reason", "detail"}
            and absences[name]["status"] == "UNAVAILABLE"
            and absences[name]["reason"] == "NOT_RECORDED_IN_DECLARED_PATH_REPORT"
            and absences[name]["detail"]
            for name in benchmark_absences
        )
        if not np.isfinite(expected.sortino):
            assert absences["sortino"]["reason"] == "ZERO_DOWNSIDE_DEVIATION"
        assert raw["selected_window_metric_absences"] == absences

        offered = activation_cli(live, capsys, "strategy-book", "controls", "--package", PACKAGE)[
            "activation"
        ]
        assert offered["next_requests"]["activate"]["task_id"] == task_id
        assert offered["review_standing"] == standing
        shown = activation_cli(live, capsys, "workspace", "show")
        (intent,) = [i for i in shown["intents"] if i.get("strategy_package_id") == PACKAGE]
        assert intent["activation"]["review_standing"] == standing
        books_request = {"operation": "CONTROLS", "strategy_package_id": PACKAGE}
        plan_request = {"operation": "RESEARCH_UPDATE_PLAN", "strategy_package_id": PACKAGE}
        first = LocalResearchClient(live.workspace)
        discovered = first.request(include_context=True)
        (forward,) = [
            row for row in discovered["intents"] if row.get("strategy_package_id") == PACKAGE
        ]
        assert forward["flow"] == "RUN_FORWARD"
        assert forward["activation"] == offered
        assert forward["activation"]["status"] == "INACTIVE"
        assert forward["next_requests"] == {"books": books_request}
        idle = first.request(plan_request)
        assert idle["status"] == "REFUSED"
        assert idle["failure_code"] == "portfolio_update.not_installed"
        assert idle["next_requests"] == {"books": books_request}
        assert idle["activation"] == offered
        assert (
            main(
                [
                    "--workspace",
                    str(live.workspace),
                    "--view",
                    "full",
                    "research-update",
                    "plan",
                    "--package",
                    PACKAGE,
                ],
                serve=lambda _: 99,
            )
            == 2
        )
        refused_cli = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert refused_cli["outcome"] == "REFUSED"
        assert refused_cli["failure_code"] == "portfolio_update.not_installed"
        assert refused_cli["detail"] == refused_cli["data"]["detail"] == idle["detail"]
        assert refused_cli["data"]["next_requests"] == {"books": books_request}
        assert refused_cli["data"]["activation"] == offered
        fresh = LocalResearchClient(live.workspace)
        book_controls = fresh.request(idle["next_requests"]["books"])
        assert book_controls["strategy_package_id"] == PACKAGE
        assert book_controls["activation"] == offered
        activate_request = book_controls["activation"]["next_requests"]["activate"]
        assert activate_request == {"operation": "STRATEGY_ACTIVATE", "task_id": task_id}
        navigation = fresh.navigation(books_request, book_controls)
        assert navigation["kind"] == "portfolio_book"
        assert parse_qs(urlsplit(navigation["url"]).fragment) == {
            "page": ["portfolio"],
            "book": [task_id],
            "follow": ["latest"],
        }
        denied = fresh.request(activate_request)
        assert denied["status"] == "REFUSED"
        assert denied["failure_code"] == "strategy_activation.human_confirmation_required"
        assert fresh.request(books_request)["activation"] == offered
        panel = _json(
            live,
            f"/api/workbench/portfolio?task_id={task_id}"
            f"&portfolio_session={book.last_session.isoformat()}",
        )
        assert panel["standing"] == raw["standing"]
        assert panel["review_standing"] == standing
        assert panel["metricAbsences"] == raw["selected_window_metric_absences"]
        for alias, name in (
            ("annual", "annualized_return"),
            ("vol", "annualized_volatility"),
            ("drawdown", "maximum_drawdown"),
            ("sharpe", "sharpe"),
            ("sortino", "sortino"),
        ):
            if name not in body["selected_window_metrics"]:
                assert panel["metrics"][alias] is None
        for alias in (
            "informationRatio",
            "benchmarkRelative",
            "beta",
            "trackingError",
            "jensenAlpha",
        ):
            assert panel["metrics"][alias] is None

        assert (
            len(live.session.task_control_registry.tasks()),
            live.review.artifacts.write_count,
        ) == (before)
        activated = _json(
            live, "/api/strategy/activate", method="POST", payload={"task_id": task_id}
        )
        assert activated["status"] == "ACTIVATED", activated
        assert activated["fit_calls"] == 0
        assert activated["review_standing"] == standing
        active = activation_cli(live, capsys, "strategy-book", "controls", "--package", PACKAGE)[
            "activation"
        ]
        assert active["status"] == "ACTIVE" and active["review_standing"] == standing
        after = LocalResearchClient(live.workspace)
        active_controls = after.request(books_request)
        assert active_controls["activation"] == active
        assert active_controls["activation"]["book_task_id"] == task_id
        reopened = after.request(include_context=True)
        (forward,) = [
            row for row in reopened["intents"] if row.get("strategy_package_id") == PACKAGE
        ]
        assert forward["activation"] == active
        assert forward["next_requests"] == {"update": plan_request}
        # The activation names the person's one-click stop beside its update.
        assert activated["next_requests"] == {
            "update": plan_request,
            "deactivate": {"operation": "STRATEGY_DEACTIVATE", "strategy_package_id": PACKAGE},
        }

        def no_review_read(*_args, **_kwargs):
            raise AssertionError("Automation discovery opened full review standing")

        with monkeypatch.context() as metadata:
            metadata.setattr(live.operations.activations, "read_review", no_review_read)
            automation = activation_cli(live, capsys, "automation", "show")
        (running,) = [r for r in automation["runs_forward"] if r["strategy_package_id"] == PACKAGE]
        assert "review_standing" not in running
        for field in (
            "status",
            "book_task_id",
            "first_forward_session",
            "horizon",
            "strategy_dates",
        ):
            assert running[field] == active[field]
        assert activation_cli(live, capsys, "result", "show", result_hash)["standing"][
            "activation"
        ] == ("ACTIVE")
        # Old completed books can retain their Task and ledger without a pipeline manifest.
        # The same activation admission still names the exact book's review in that case.
        lineage = live.application.pipeline.content.root.resolve()
        retained = lineage.with_name("pipeline-not-retained")
        assert lineage.is_relative_to(live.workspace.resolve())
        assert retained.is_relative_to(live.workspace.resolve())
        lineage.rename(retained)
        assert live.application.pipeline.find_for_task(UUID(task_id)) is None
        historical = activation_cli(
            live, capsys, "strategy-book", "controls", "--package", PACKAGE
        )["activation"]
        assert historical["review_standing"] == standing
        historical_report = activation_cli(live, capsys, "result", "show", result_hash)
        assert historical_report["review_standing"] == standing
        assert historical_report["standing"]["activation"] == "ACTIVE"


def test_an_absent_or_inactive_book_keeps_its_selected_way_on(live):
    """An absent or inactive book keeps the selected package in its way on."""
    absent = _json(live, "/api/strategy/activate", method="POST", payload={"task_id": str(uuid4())})
    assert absent["status"] == "REFUSED", absent
    assert absent["failure_code"] == "strategy_activation.book_task_absent"
    assert absent["detail"] and absent["next_requests"] == {"books": {"operation": "CONTROLS"}}
    idle = _json(
        live,
        "/api/strategy/deactivate",
        method="POST",
        payload={"strategy_package_id": "RETURN_G6_MU_ONLY"},
    )
    assert idle["status"] == "REFUSED" and idle["failure_code"] == "strategy_activation.not_active"
    package = {"operation": "CONTROLS", "strategy_package_id": "RETURN_G6_MU_ONLY"}
    assert idle["next_requests"] == {"books": package}
    code, scored, _ = _cli(live.workspace, "score", "plan", "--package", "RETURN_G6_MU_ONLY")
    assert code == 2 and scored["next_requests"]["books"] == package, scored
    assert "--package RETURN_G6_MU_ONLY" in scored["next_commands"]["books"]
