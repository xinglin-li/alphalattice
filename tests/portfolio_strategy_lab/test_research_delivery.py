"""Joined delivery boundaries over existing verified-readback ports; no numerical work."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from typing import get_args
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
    # A replay is held current through its estimators' recorded moves.
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
    with pytest.raises(ValueError, match=r"research_delivery\.subject_required"):
        export_research_delivery(
            request=PortfolioResearchOperationRequest(
                operation="EXPERIMENT_DELIVERY_EXPORT", task_id=primary
            ),
            experiments=app,
            review=None,
            caller="EXTERNAL_AUTOMATION",
        )
    with pytest.raises(ValueError, match="comparison_pair_required"):
        PortfolioResearchOperationRequest(
            operation="EXPERIMENT_DELIVERY_EXPORT", task_id=primary, left_task_id=other
        )


def test_delivery_renderer_keeps_units_gaps_and_escapes_commentary(monkeypatch):
    from alphalattice.interface.local_application import experiment_report
    from alphalattice.interface.local_application.cli_contract import ANSWER_LANGUAGE
    from alphalattice.investment.portfolio_strategy_lab.reporting.static import (
        DECISION_WORDS,
        format_book_weight,
    )

    portfolio = _reports()[2][UUID(int=1)]
    portfolio["portfolio_source"]["target_recipe_id"] = "target"
    portfolio["position"].update(targets=[0.02], weights=[0.00700001349], cash=0.0032)
    portfolio["document"]["portfolio"] = {"concentration": 0.123456789}
    portfolio["execution_preview"] = {"score_session_count": 1, "hold_session_count": 0}
    portfolio["series"] = [
        {
            "session": "2024-08-12",
            "decision_mode": "HOLD",
            "gross_simple_return": 0.032,
            "net_simple_return": 0.0123456789,
            "benchmark_simple_return": -0.003,
            "one_way_turnover": 0.02,
            "cost_fraction": 0.000034567,
        }
    ]
    portfolio["result"].update(sharpe=0.0000012345, sortino=0.123456789)
    factor_classes = {
        "POSITIVE_OOS_EVIDENCE": "Positive out-of-sample evidence",
        "MIXED_OOS_EVIDENCE": "Mixed out-of-sample evidence",
        "NO_DETECTABLE_EFFECT": "No detectable effect",
        "NEGATIVE_OOS_EVIDENCE": "Negative out-of-sample evidence",
        "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
    }
    factor_reasons = (
        "DIRECTIONALLY_POSITIVE_BY_CONFIRMED",
        "DIRECTIONALLY_POSITIVE_NOT_BY_CONFIRMED",
        "DIRECTIONALLY_NEGATIVE_BY_CONFIRMED",
        "NONPOSITIVE_DIRECTION_NOT_BY_CONFIRMED",
        "MIXED_OOS_DIRECTION",
    )
    factor = _reports()[2][UUID(int=3)]
    factor["document"] = {"factor": {"factor_ids": [f"factor-{i}" for i in range(7)]}}
    factor["result"] = {
        "evidence_report": {
            "items": [
                {
                    "factor_id": f"factor-{i}",
                    "classification": classification,
                    "reason_codes": [reason],
                }
                for i, (classification, reason) in enumerate(
                    zip(
                        (
                            "POSITIVE_OOS_EVIDENCE",
                            "POSITIVE_OOS_EVIDENCE",
                            "NEGATIVE_OOS_EVIDENCE",
                            "NO_DETECTABLE_EFFECT",
                            "MIXED_OOS_EVIDENCE",
                            "INSUFFICIENT_EVIDENCE",
                            "OWNER_FUTURE_CLASS",
                        ),
                        (*factor_reasons, "OWNER_FUTURE_REASON", "OWNER_FUTURE_REASON"),
                        strict=True,
                    )
                )
            ]
        }
    }
    monkeypatch.setattr(
        experiment_report,
        "alpha_report_section",
        lambda _: '<h1>Alpha</h1><section id="study-context"></section>',
    )
    body = {
        "claim": "No winner",
        "question": None,
        "question_status": "NOT_RECORDED",
        "input": {
            "operation": "EXPERIMENT_DELIVERY_EXPORT",
            "position_basis": "CONDITIONAL_ESTIMATE",
            "audit_boundary": "POST_OBSERVED_QA_NOT_TIMELY_ADVICE",
            "future": {"availability": "OWNER_FUTURE_CODE", "value": None},
        },
        "sections": {
            "portfolio": {"value": portfolio},
            "factor": {"value": factor},
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
                                    "left": 10.23456789123,
                                    "right": None,
                                }
                            ],
                        }
                    ],
                },
            },
        },
        "commentary_provenance": {"submitted_by": "HOST", "attribution": "COMMITTEE_FLOOR"},
        "commentary": [
            {"attribution": "<script>actor</script>", "text": "<script>alert(1)</script>"},
            {
                "attribution_word": "Committee verdict: {outcome}",
                "attribution_words": {"outcome": "PROCEED_WITH_NOTES"},
                "text": "kept verdict",
            },
            *(
                {
                    "attribution_word": "{member}, final view",
                    "attribution_words": {"member": role},
                    "text": "kept view",
                }
                for role in ("ALPHA", "RISK")
            ),
            {
                "attribution": "Person",
                "text": "unused composed text",
                "question": "Kept question?",
                "relay_word": "Relayed by the agent",
                "answer_present": True,
                "answer": "Keep 5%.",
            },
        ],
    }
    sealed = deepcopy(body)
    rendered = experiment_report.render_research_delivery(body)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "bps on platform one-way turnover" in rendered and "Not available" in rendered
    for code, words in {
        "NOT_SELECTED": "Not selected",
        "NOT_RECORDED": "Not recorded",
        "HOST": "Host",
        "COMMITTEE_FLOOR": "Committee floor",
        "ALPHA": "Alpha",
        "RISK": "Risk",
        "EXPERIMENT_DELIVERY_EXPORT": "Delivery export",
        "CONDITIONAL_ESTIMATE": "Conditional estimated weights",
        "POST_OBSERVED_QA_NOT_TIMELY_ADVICE": DECISION_WORDS["POST_OBSERVED_QA_NOT_TIMELY_ADVICE"],
        "HOLD": "hold",
        "COST": "Cost",
    }.items():
        assert code not in rendered and words in rendered
    assert "OWNER_FUTURE_CODE" in rendered
    assert (
        body["commentary"][1]["attribution_word"].format(outcome="Proceed with notes") in rendered
    )
    assert body["commentary"][-1]["relay_word"] in rendered
    for code, words in factor_classes.items():
        assert code not in rendered and words in rendered
    assert all(reason not in rendered for reason in factor_reasons)
    assert all(
        term in rendered
        for term in (
            "Positive in direction",
            "Negative in direction",
            "Not positive in direction",
            "Mixed direction",
        )
    )
    assert rendered.count("Benjamini\u2013Yekutieli FDR") == 4
    assert "OWNER_FUTURE_CLASS" in rendered and "OWNER_FUTURE_REASON" in rendered
    for section, code, words in (
        ("foundation", "PRESENT", "present"),
        ("comparison", "INCOMPATIBLE", "refused by the owner"),
        ("evidence_cro", "HISTORICAL_REVIEW", "historical review"),
        ("evidence_cro", "EXPIRED", "review present but expired"),
    ):
        state = deepcopy(body)
        state["sections"][section] = {"status": code}
        assert f">{words}<" in experiment_report.render_research_delivery(state)
    state = deepcopy(body)
    state["sections"]["portfolio"]["value"]["series"][0]["decision_mode"] = "REBALANCE"
    assert ">rebalance<" in experiment_report.render_research_delivery(state)
    for code, words in (
        ("HOLDINGS", "Holdings"),
        ("CONCENTRATION", "Concentration"),
        ("TURNOVER", "Turnover"),
        ("PERFORMANCE", "Performance"),
    ):
        state = deepcopy(body)
        state["sections"]["comparison"]["value"]["dimensions"][0]["dimension"] = code
        assert f">{words}<" in experiment_report.render_research_delivery(state)
    for value in (0.00700001349, 0.123456789, 0.0123456789, 10.23456789123, 0.0000012345):
        assert str(value) not in rendered
    assert all(
        f">{figure}<" in rendered
        for figure in ("0.70", "2.00", "3.20", "1.23", "10.23", "0.1235", "1.23 \u00d7 10^-6")
    )
    assert format_book_weight(portfolio["position"]["cash"], precision=2) in rendered
    assert 'id="factor-study-context"' in rendered and 'id="alpha-study-context"' in rendered
    assert rendered.count("<h1>") == 1
    for language in ("en", "zh"):
        token = ANSWER_LANGUAGE.set(language)
        try:
            assert experiment_report.render_research_delivery(body) == rendered
            assert ANSWER_LANGUAGE.get() == language
            assert "PROCEED_WITH_NOTES" not in rendered
            assert "Keep 5%." in rendered and "unused composed text" not in rendered
        finally:
            ANSWER_LANGUAGE.reset(token)
    assert body == sealed


def test_each_explained_refusal_keeps_its_code_and_asks_only_operations_the_host_accepts():
    """Binding plan B9: a refusal a person meets after an upgrade says what happened and what
    to request next; the code stays exact for an Agent, and a next request the Host would
    refuse would send the person from one refusal into another."""

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
        # a plan refusal names its field and the rule it broke.
        "factor_research.handoff_authority_field_mismatch:experiment.universe_handle",
        "alpha_research.qualification_nomination_invalid:not_in_family",
        "alpha_research.qualification_unexpected_parent:factor_task_id",
        # refusals name their field and the way on.
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


@pytest.mark.parametrize(
    "code",
    [
        "strategy_book.strategy_package_required",
        "local_application.strategy_package_not_installed:NOT_AN_INSTALLED_PACKAGE",
        "research_workspace.strategy_not_installed",
    ],
)
def test_every_book_selector_refusal_has_door_words_and_owner_context(code) -> None:
    """Selector refusals keep door keys and bound controls for each installed choice."""
    from alphalattice.control.product_host.composition.plain_refusals import explain
    from alphalattice.interface.local_application.cli_contract import refusal_words, request_problem

    words = refusal_words(code)
    assert set(words) == {"detail", "next_action"} and all(words.values())
    packages = ("INSTALLED_A", "INSTALLED_B")
    answer = explain(code, installed_packages=packages)
    assert answer["detail"] and answer["next_action"]
    for request in answer["next_requests"].values():
        assert request_problem(request) is None
    if code != "research_workspace.strategy_not_installed":
        assert answer["next_requests"] == {
            f"controls:{package}": {"operation": "CONTROLS", "strategy_package_id": package}
            for package in packages
        }
        assert all(package in answer["detail"] for package in packages)
