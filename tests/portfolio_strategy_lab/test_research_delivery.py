"""Joined delivery boundaries over existing verified-readback ports; no numerical work."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import pytest

from alphalattice.control.product_host.composition.research_delivery import export_research_delivery
from alphalattice.interface.local_application.portfolio_research import (
    PortfolioResearchOperationRequest,
    PortfolioResearchRequestDocument,
)


def test_readback_route_keeps_current_replay_and_foundation_admission_strict(monkeypatch, tmp_path):
    from alphalattice.control.product_host.composition.research_experiments import (
        ResearchExperimentApplication,
    )

    app = object.__new__(ResearchExperimentApplication)
    app.session = SimpleNamespace(
        workspace=tmp_path, task_control_registry=SimpleNamespace(task=lambda _: "task")
    )
    calls = []
    monkeypatch.setattr(app, "_of", lambda _: ("plan", None))
    monkeypatch.setattr(app, "_current", lambda p, **kw: calls.append((p, kw)))
    monkeypatch.setattr(
        app,
        "readback",
        lambda *a, **kw: {"status": "CAPTURED", "current_policy": kw["current_policy"]},
    )
    for operation, expected in (("EXPERIMENT_READBACK", False), ("EXPERIMENT_REPLAY", True)):
        response = app.operate(
            PortfolioResearchOperationRequest(operation=operation, task_id=UUID(int=1)),
            caller="EXTERNAL_AUTOMATION",
        )
        assert response == {"status": "CAPTURED", "current_policy": expected}
    # A replay is held current through its estimators' recorded moves (V314).
    assert calls == [("plan", {"replay": True})]
    with pytest.raises(TypeError):
        PortfolioResearchOperationRequest(
            operation="EXPERIMENT_REPLAY", task_id=UUID(int=1), recorded_readback=True
        )

    def foundation(_hash, *, recorded_readback=False):
        calls.append(recorded_readback)
        raise ValueError("research_foundation.admission_artifact_unavailable")

    monkeypatch.setattr(app, "_foundation", foundation)
    for operation, expected in (
        ("EXPERIMENT_FOUNDATION_READBACK", True),
        ("EXPERIMENT_FOUNDATION_EXPORT", True),
        ("EXPERIMENT_FOUNDATION_DRAFT", False),
    ):
        response = app.operate(
            PortfolioResearchOperationRequest(
                operation=operation, foundation_admission_hash="a" * 64
            ),
            caller="EXTERNAL_AUTOMATION",
        )
        assert response == {
            "status": "REFUSED",
            "foundation_admission_hash": "a" * 64,
            "failure_code": "research_foundation.admission_artifact_unavailable",
            "numerical_call_count": 0,
            "detail": (
                "The admission's recorded sources could not be verified. Read Research "
                "inputs to plan a new Factor study."
            ),
            "next_action": "RESEARCH_INPUTS",
            "next_requests": {"inputs": {"operation": "RESEARCH_INPUTS"}},
        }
        assert calls[-1] is expected


def _reports():
    primary, alpha, factor, other = (UUID(int=i) for i in range(1, 5))
    source = {
        "alpha_task_id": str(alpha),
        "alpha_receipt_hash": "a" * 64,
        "alpha_program_hash": "b" * 64,
        "candidate_id": "candidate",
        "input_binding_hash": "c" * 64,
        "panel_snapshot_hash": "d" * 64,
        "outcome_snapshot_hash": "e" * 64,
        "universe_revision": "universe",
        "ordered_listing_ids": ["listing"],
        "formation_sessions": ["2024-08-12"],
    }
    position = {
        "session": "2024-08-12",
        "holding_count": 1,
        "cash": 0.0,
        "hhi": 1.0,
        "one_way_turnover": 0.1,
        "cost_fraction": 0.001,
        "gross_simple_return": 0.1,
        "net_simple_return": 0.099,
    }
    book = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": str(primary),
        "portfolio_source": source,
        "research_input_id": "research",
        "input_binding_hash": "c" * 64,
        "receipt": {"receipt_hash": "f" * 64, "market_binding": "market"},
        "position": position,
        "result": {"sharpe": None, "cost_bps": 10},
        "document": {},
        "series": [],
        "limitations": ["Observed, not independent."],
    }
    alternate = deepcopy(book)
    alternate["task_id"] = str(other)
    alternate["receipt"]["receipt_hash"] = "1" * 64
    return (
        primary,
        other,
        {
            primary: book,
            other: alternate,
            alpha: {
                "status": "EXPERIMENT_PUBLISHED",
                "receipt": {"receipt_hash": "a" * 64},
                "program": {"program_hash": "b" * 64},
                "input_binding_hash": "c" * 64,
                "result": {"candidates": [{"candidate_id": "candidate"}]},
                "alpha_source": {
                    "factor_task_id": str(factor),
                    "factor_receipt_hash": "2" * 64,
                    "factor_binding": {"binding_hash": "c" * 64},
                },
            },
            factor: {
                "status": "EXPERIMENT_PUBLISHED",
                "receipt": {"receipt_hash": "2" * 64},
                "input_binding_hash": "c" * 64,
            },
        },
    )


def test_delivery_preserves_sources_attribution_partial_states_and_revalidates(monkeypatch):
    primary, other, reports = _reports()
    counts = Counter()

    def read(task_id, session=None):
        counts[task_id] += 1
        return deepcopy(reports[task_id])

    app = SimpleNamespace(readback=read)
    monkeypatch.setattr(
        "alphalattice.interface.local_application.experiment_report.render_research_delivery",
        lambda value: "<html>presentation only</html>",
    )
    document = PortfolioResearchRequestDocument(
        operation="EXPERIMENT_DELIVERY_EXPORT",
        task_id=primary,
        experiment_receipt_hash="f" * 64,
        portfolio_session="2024-08-12",
        left_task_id=other,
        right_task_id=primary,
        delivery_question="What does this comparison establish?",
        delivery_commentary=[{"attribution": "PM / external", "text": "No winner established."}],
    )
    before = deepcopy(reports)
    export = export_research_delivery(
        request=document.to_operation_request(),
        experiments=app,
        review=None,
        caller="EXTERNAL_AUTOMATION",
    )
    assert reports == before
    assert export["sections"]["portfolio"]["value"] == reports[primary]
    assert (
        export["sections"]["comparison"]["value"]["disposition"]
        == "DECLARED_PATH_COMPARISON_NO_SELECTION"
    )
    assert (
        export["sections"]["risk"]["status"]
        == export["sections"]["evidence_cro"]["status"]
        == "NOT_SELECTED"
    )
    assert export["commentary"][0]["text"] == "No winner established."
    assert export["commentary_provenance"]["submitted_by"] == "EXTERNAL_AUTOMATION"
    assert counts[primary] == 1
    assert export["sections"]["comparison"]["value"]["left"]["task_id"] == str(other)
    assert export["sections"]["comparison"]["value"]["right"]["task_id"] == str(primary)
    assert (
        PortfolioResearchRequestDocument.model_validate(
            export["next_requests"]["reopen"]
        ).to_operation_request()
        == document.to_operation_request()
    )
    reports[other]["portfolio_source"]["input_binding_hash"] = "0" * 64
    partial = export_research_delivery(
        request=document.to_operation_request(),
        experiments=app,
        review=None,
        caller="EXTERNAL_AUTOMATION",
    )
    assert partial["sections"]["comparison"]["status"] == "INCOMPATIBLE"
    assert "value" not in partial["sections"]["comparison"]
    reports[primary]["receipt"]["receipt_hash"] = "0" * 64
    with pytest.raises(ValueError, match="portfolio_subject_mismatch"):
        export_research_delivery(
            request=document.to_operation_request(),
            experiments=app,
            review=None,
            caller="EXTERNAL_AUTOMATION",
        )
    with pytest.raises(ValueError, match="operation_field_required"):
        PortfolioResearchOperationRequest(operation="EXPERIMENT_DELIVERY_EXPORT", task_id=primary)
    with pytest.raises(ValueError, match="comparison_pair_required"):
        PortfolioResearchOperationRequest(
            operation="EXPERIMENT_DELIVERY_EXPORT", task_id=primary, left_task_id=other
        )


def test_delivery_renderer_keeps_units_gaps_and_escapes_commentary(monkeypatch):
    from alphalattice.interface.local_application import experiment_report

    monkeypatch.setattr(
        experiment_report, "_portfolio_report", lambda _: "<html><h1>Portfolio</h1></html>"
    )
    monkeypatch.setattr(
        experiment_report,
        "factor_report_section",
        lambda _: '<h1>Factor</h1><section id="study-context"></section>',
    )
    monkeypatch.setattr(
        experiment_report,
        "alpha_report_section",
        lambda _: '<h1>Alpha</h1><section id="study-context"></section>',
    )
    body = {
        "claim": "No winner",
        "question": None,
        "question_status": "NOT_RECORDED",
        "input": {},
        "sections": {
            "portfolio": {"value": {}},
            "factor": {"value": {}},
            "alpha": {"value": {}},
            "foundation": {"status": "NOT_PUBLISHED"},
            "risk": {"status": "NOT_SELECTED"},
            "evidence_cro": {"status": "NOT_SELECTED"},
            "comparison": {
                "status": "PRESENT",
                "value": {
                    "claim": "No selection",
                    "left": {"task_id": "left"},
                    "right": {"task_id": "right"},
                    "dimensions": [
                        {
                            "dimension": "COST",
                            "metrics": [
                                {
                                    "label": "platform cost",
                                    "unit": "bps on platform one-way turnover",
                                    "left": 10,
                                    "right": None,
                                }
                            ],
                        }
                    ],
                },
            },
        },
        "commentary_provenance": {"attribution": "CALLER_SUPPLIED"},
        "commentary": [
            {"attribution": "<script>actor</script>", "text": "<script>alert(1)</script>"}
        ],
    }
    rendered = experiment_report.render_research_delivery(body)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "bps on platform one-way turnover" in rendered and "Not available" in rendered
    assert "NOT_SELECTED" in rendered and "NOT_RECORDED" in rendered
    assert 'id="factor-study-context"' in rendered and 'id="alpha-study-context"' in rendered
    assert rendered.count("<h1>") == 1


def test_each_explained_refusal_keeps_its_code_and_asks_only_operations_the_host_accepts():
    """Binding plan B9: a refusal a person meets after an upgrade says what happened and what
    to request next; the code stays exact for an Agent, and a next request the Host would
    refuse would send the person from one refusal into another."""

    from typing import get_args

    from alphalattice.capabilities.alpha_modeling.runtime.lightgbm_threads import (
        LightGBMThreadCanaryMismatch,
    )
    from alphalattice.control.product_host.composition import plain_refusals
    from alphalattice.interface.local_application.portfolio_research import (
        PortfolioResearchOperation,
    )

    operations = set(get_args(PortfolioResearchOperation.__value__))
    codes = (
        *sorted(plain_refusals._CHANGED),
        "research_lane.exploration_not_promoted:00000000-0000-0000-0000-000000000001",
        "research_lane.sample_size_invalid",
        "research_lane.sample_not_supported",
        "task_control.task_not_found",
        "research_experiment.summary_kind_not_installed",
        "research_workspace.strategy_not_installed",
        "portfolio_research.result_not_found",
        "content_store.artifact_tampered",
        "portfolio_application.session_outside_report",
        "local_application.task_recovery_not_configured",
        LightGBMThreadCanaryMismatch.code,
        "task_not_succeeded",
        "product_host.evidence_review_evidence_not_current",
        # V292: a plan refusal names its field and the rule it broke.
        "factor_research.handoff_authority_field_mismatch:experiment.universe_handle",
        "alpha_research.qualification_nomination_invalid:not_in_family",
        "alpha_research.qualification_unexpected_parent:factor_task_id",
        # V318, V320: AX6's refusals name their field and the way on.
        "research_experiment.standalone_kind_not_installed",
        "goal.continuation_requires_parent_goal",
        "workspace_data_update.panel_binding_mismatch",
    )
    context = {
        "task_id": "t",
        "kind": "Factor",
        "refresh_calls": 3,
        "lifecycle": "FAILED",
        "result_hash": "h",
    }
    for code in codes:
        body = plain_refusals.refused(code, **context)
        assert body["status"] == "REFUSED" and body["failure_code"] == code and body["detail"]
        asked = {request["operation"] for request in body["next_requests"].values()}
        assert asked <= operations, (code, asked - operations)

    portfolio = plain_refusals.explain("research_experiment.standalone_kind_not_installed")
    assert portfolio["fields"] == [["experiment_kind"]]
    draft = portfolio["next_requests"]["portfolio_draft"]
    assert draft["operation"] == "EXPERIMENT_PORTFOLIO_DRAFT"
    continuing = plain_refusals.explain("goal.continuation_requires_parent_goal")
    assert continuing["fields"] == [["research", "purpose"], ["parent_goal_id"]]
    changed = plain_refusals.explain("research_experiment.execution_binding_changed", **context)
    assert changed["detail"].startswith("This Factor study cannot be replayed as it ran")
    assert "3 numerical calls" in changed["detail"]
    assert set(changed["next_requests"]) == {"readback", "continue", "overview"}
    stale = plain_refusals.explain(
        "product_host.evidence_review_evidence_not_current", book={"result_hash": "h"}
    )
    assert stale["next_requests"] == {"evidence": {"operation": "EVIDENCE_CRO", "result_hash": "h"}}
    panel_code = "workspace_data_update.panel_binding_mismatch"
    panel = plain_refusals.refused(panel_code, **context)
    assert panel["status"] == "REFUSED" and panel["failure_code"] == panel_code
    assert panel["detail"] == (
        "The current Panel does not match the data binding required for this update. "
        "Read the data update's current record and the input state it reports; resolve "
        "that condition before planning the update again."
    )
    assert panel["next_action"] == "DATA_UPDATE_READBACK"
    assert panel["next_requests"] == {"show": {"operation": "DATA_UPDATE_READBACK"}}
    assert plain_refusals.explain("an.unlisted_code") == {}
