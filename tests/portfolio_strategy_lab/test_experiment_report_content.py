"""Presentation-only boundaries; no data fixture, model or database rebuild."""

from copy import deepcopy

from alphalattice.interface.local_application.evidence_cro import render_review_export
from alphalattice.interface.local_application.experiment_report import render_experiment_report


def test_report_context_preserves_support_zero_missing_and_untrusted_text():
    body = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": "<script>bad()</script>",
        "research_input_id": "chosen-input",
        "input_binding_hash": "binding",
        "program": {"kind": "factor.screening-development"},
        "document": {
            "experiment": {
                "kind": "factor.screening-development",
                "sessions": {
                    "start": "2016-01-01",
                    "end": "2026-01-01",
                    "as_of": {"session": "2026-01-02", "phase": "OFFICIAL_CLOSE"},
                },
            },
            "factor": {"factor_ids": ["f"]},
        },
        "evidence": {"formation_sessions": ["2019-01-01", "2024-01-01"], "numerical_call_count": 0},
        "result": {
            "evidence_report": {
                "items": [{"factor_id": "f", "mean_oriented_rank_ic": 0}],
                "hypothesis_count": 53,
            }
        },
        "next_requests": {"curation": {"operation": "EXPERIMENT_CURATION"}},
    }
    before = deepcopy(body)
    rendered = render_experiment_report(body)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "<dt>Requested start</dt><dd>2016-01-01</dd>" in rendered
    assert "<dt>Evaluated start</dt><dd>2019-01-01</dd>" in rendered
    assert "<dt>Numerical calls in this read</dt><dd>0</dd>" in rendered
    assert "<td>Not available</td>" in rendered and "EXPERIMENT_CURATION" in rendered
    assert body == before


def test_unpublished_task_export_is_not_an_empty_factor_result():
    rendered = render_experiment_report({"status": "RUNNING", "task_id": "pending"})
    assert "No published numerical result" in rendered and "RUNNING" in rendered
    assert "Selected Factor evidence" not in rendered


def test_review_export_keeps_partial_coverage_unadjudicated_findings_and_exact_citations_visible():
    snapshot = {
        "claim": "Historical readback only",
        "review": {
            "receipt": {
                "actor_submission": {"actor_kind": "EXTERNAL_AUTOMATION", "actor_id": "qa"},
                "model_call_count": 0,
            },
            "recommendation": {
                "route": "REQUEST_EVIDENCE_REFRESH",
                "review_state": "PARTIAL",
                "issuer_conclusions": [
                    {
                        "entity_id": "A",
                        "conclusion": "CONCERNS_NOT_ADJUDICATED",
                        "finding_count": 1,
                        "adverse_issue_count": 0,
                    }
                ],
                "reasons": [],
                "required_actions": [],
            },
            "dossier": {
                "coverage": {
                    "mapping_coverage": 0.02,
                    "selected_issuer_coverage": 1.0,
                    "unavailable_reasons": ["Most of the book has no admitted evidence"],
                },
                "limitations": ["Not independent textual entailment proof"],
                "findings": [
                    {
                        "finding_handle": "F-A",
                        "summary": "<script>finding</script>",
                        "topic": "LEGAL",
                        "direction": "ADVERSE",
                        "structure": "SUPPORTED",
                        "supporting_span_handles": ["S-A"],
                        "contradicting_span_handles": ["S-B"],
                    }
                ],
                "citations": [
                    {
                        "span_handle": "S-A",
                        "document_handle": "D-A",
                        "entity_id": "A",
                        "available_at": "2026-01-01",
                    }
                ],
            },
        },
        "evidence": {
            "verified_spans": [
                {
                    "span_handle": "S-A",
                    "character_start": 10,
                    "character_end": 25,
                    "start_line": 2,
                    "end_line": 3,
                    "excerpt": "<p>source</p>",
                }
            ]
        },
    }
    before = deepcopy(snapshot)
    rendered = render_review_export(snapshot, "<html><h1>Original book</h1></html>")
    assert "Original book" in rendered
    assert rendered.index("Linked recommendation review") < rendered.index("Original book")
    assert "Review completeness: PARTIAL" in rendered
    assert "Whole-book mapped ending weight</dt><dd>2.000%" in rendered
    assert "Selected-issuer scope reviewed coverage</dt><dd>100.000%" in rendered
    assert "adjudication not completed" in rendered and "adverse issues recorded: 0" not in rendered
    assert 'href="#citation-S-A"' in rendered and 'id="citation-S-A"' in rendered
    assert "Contrary citations:" in rendered and "S-B" in rendered
    assert 'href="#citation-S-B"' not in rendered
    assert "S-B (citation record unavailable)" in rendered
    assert "LEGAL / lifecycle not recorded / ADVERSE" in rendered
    assert "<script>" not in rendered and "&lt;script&gt;finding" in rendered
    assert "&lt;p&gt;source&lt;/p&gt;" in rendered
    assert "Most of the book has no admitted evidence" in rendered and snapshot == before


def test_a_qualification_study_reports_its_conclusion_not_a_development_result():
    """regression (the fork's baseline): a family qualification's export answered
    `research_experiment.refused:KeyError:<n>`, since every Alpha study went to the development
    section, which reads a result a qualification does not hold; it reports its own conclusion."""
    body = {
        "status": "EXPERIMENT_PUBLISHED",
        "task_id": "qualification-task",
        "program": {"kind": "alpha.model-development"},
        "document": {"experiment": {"kind": "alpha.model-development"}},
        "execution_preview": {},
        "alpha_qualification": {
            "disposition": "NO_STABLE_CURRENT_ALPHA_MODEL",
            "attempted_candidate_ids": ["a", "b", "c"],
            "selected_candidate_ids": [],
            "current_qualified_ids": [],
            "fit_call_count": 6,
            "predict_call_count": 6,
            "qualification_hash": "f" * 64,
        },
        "next_requests": {},
    }
    rendered = render_experiment_report(body)
    assert "Alpha family qualification" in rendered
    assert "<dt>Disposition</dt><dd>NO_STABLE_CURRENT_ALPHA_MODEL</dd>" in rendered
    assert "<dt>Attempted candidates</dt><dd>3</dd>" in rendered
