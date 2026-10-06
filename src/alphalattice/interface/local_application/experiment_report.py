"""Readable export of an already verified experiment projection; no calculations."""

from __future__ import annotations

import html
import json
from typing import Any


def _text(value: object) -> str:
    """Escape owner values consistently; unavailable is never a numeric zero."""
    if isinstance(value, dict) and "availability" in value:
        value = value.get("value") if value.get("value") is not None else value["availability"]
    if isinstance(value, dict | list):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return html.escape("Not available" if value is None else str(value))


def _facts(values: dict[str, Any]) -> str:
    return (
        "<dl>"
        + "".join(
            f"<dt>{_text(label)}</dt><dd>{_text(value)}</dd>" for label, value in values.items()
        )
        + "</dl>"
    )


def render_goal(body: dict[str, Any]) -> str:
    """A goal's reading surface: its declaration, evidence, statements and end; no winner."""
    goal = body["goal"]
    declaration = goal["declaration"]
    parts = [
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Goal</title><style>body{font:16px system-ui;margin:2rem auto;"
        "max-width:1100px;padding:1rem}pre{white-space:pre-wrap}dd,pre,p{overflow-wrap:anywhere}"
        "article{border-top:1px solid #aaa;margin:1rem 0;padding:1rem 0}</style><body>",
        f"<h1>{_text(declaration['title'])}</h1><p>{_text(declaration['objective'])}</p>",
        _facts(
            {
                "State": goal["state"],
                **{
                    k.replace("_", " ").capitalize(): declaration[k]
                    for k in (
                        "kind",
                        "scope",
                        "criteria",
                        "deliverables",
                        "constraints",
                        "research",
                    )
                    if declaration.get(k)
                },
            }
        ),
        "<p>" + _text(body["claim"]) + "</p><h2>Evidence and next choices</h2>",
    ]
    for row in body["references"]:
        ref = row["reference"]
        parts.extend(
            [
                f"<article><h3>{_text(ref['label'])}</h3>",
                _facts(
                    {"Stage": ref["stage"], "State": row["state"], "Intent": ref["intent_relation"]}
                ),
                _facts(row.get("summary", {"Failure": row.get("failure_code")})),
                "<details><summary>Exact reference and legal next requests</summary>"
                + _facts({"Reference": ref, "Next": row["next_requests"]})
                + "</details></article>",
            ]
        )
    parts.append("<h2>Conclusions, disagreements and PM response</h2>")
    for item in goal["statements"]:
        parts.append(
            f"<article><h3>{_text(item['attribution'])} · {_text(item['disposition'])}</h3>"
            f"<p>{_text(item['text'])}</p>"
            + _facts({"Evidence": item["evidence"], "Response to": item["responds_to"]})
            + _facts(
                {"Objective revision": body.get("statement_context", {}).get(item["statement_id"])}
            )
            + "</article>"
        )
    if not goal["statements"]:
        parts.append("<p>No conclusion recorded; completed computation is not an answer.</p>")
    record = body.get("record") or {}
    if record.get("conversation"):
        # The Team's messages made under the goal: declarations, each with its sender (GR2).
        parts.append("<h2>Conversation</h2>")
        for message in record["conversation"]:
            recipient = f" → {_text(message['recipient_id'])}" if message["recipient_id"] else ""
            parts.append(
                f"<article><h3>{_text(message['message_kind'])} · "
                f"{_text(message['agent_id'])}{recipient}</h3>"
                f"<p>{_text(message['summary'])}</p>"
                + _facts({"Reply to": message["reply_to"], "Packet": message.get("packet_hash")})
                + "</article>"
            )
        parts.append(
            _facts(
                {
                    "Messages": record["message_count"],
                    "Open assignments": record["open_assignments"],
                }
            )
        )
    if goal.get("submission"):
        submission = goal["submission"]
        parts.extend(
            [
                "<h2>Submission</h2>",
                f"<p>{_text(submission['summary'])}</p>",
                _facts(
                    {
                        "Outcome": submission["outcome"],
                        "Criteria": submission["criteria"],
                        "Findings": submission["findings"],
                        "Problems": submission["problems"],
                        "Follow-ups": submission["follow_ups"],
                        "The Host's record": goal["completion"],
                    }
                ),
            ]
        )
    parts.extend(
        [
            "<h2>Unresolved evidence</h2>",
            _facts({"Gaps": body["gaps"], "Open choices": body["open_choices"]}),
            "<details><summary>Exact identity, revision and declaration</summary>",
            _facts(goal),
            "</details></body></html>",
        ]
    )
    return "".join(parts)


def _method_facts(body: dict[str, Any]) -> dict[str, Any]:
    """The installed-method verdict, shown only for a Desk whose verifier states one."""
    if body.get("method_standing") is None:
        return {}
    refusal = body.get("method_refusal")
    return {
        "Method in this version": body["method_standing"],
        **({"Why a run would refuse it": refusal} if refusal else {}),
    }


def _study_context(body: dict[str, Any]) -> str:
    """Display the recorded study and its inputs, not today's mutable workspace."""
    experiment = body.get("document", {}).get("experiment", {})
    declared = experiment.get("sessions", {})
    evidence = body.get("evidence", {})
    sessions = evidence.get("formation_sessions") or []
    facts = {
        "Task": body.get("task_id"),
        "Research kind": experiment.get("kind", body.get("program", {}).get("kind")),
        "Research input": body.get("research_input_id"),
        "Input binding": body.get("input_binding_hash"),
        "Data snapshot": experiment.get("data_snapshot_handle"),
        "Universe declaration": experiment.get("universe_handle"),
        "Requested start": declared.get("start"),
        "Requested end": declared.get("end"),
        "Knowledge cutoff": declared.get("as_of"),
        "Evaluated start": sessions[0] if sessions else None,
        "Evaluated end": sessions[-1] if sessions else None,
        "Evaluated formation count": len(sessions) if sessions else None,
        "Publication intent": experiment.get("publication_intent"),
        "Evidence verification": body.get("evidence_verification"),
        "Readback disposition": evidence.get("disposition"),
        **_method_facts(body),
        "Numerical calls in completing execution": body.get("execution_numerical_call_count"),
        "Original execution evidence": body.get("execution_evidence_hash"),
        "Numerical calls in this read": evidence.get("numerical_call_count"),
    }
    return (
        '<section id="study-context"><h2>Study context and provenance</h2>'
        "<p>Requested data support and evaluated support are different. The declared "
        "cutoff is not the time this page was opened. Readback counts are not the "
        "original execution's work. Missing metadata is unavailable, not inferred.</p>"
        + _facts(facts)
        + research_timing_section(body.get("timing"))
        + "<details><summary>Declared budget and reproducibility</summary>"
        + _facts({"Budget": experiment.get("budget"), "Determinism": experiment.get("determinism")})
        + "</details></section>"
    )


def research_timing_section(timing: dict[str, Any] | None) -> str:
    """Render the shared readback; never infer a timestamp from a date."""
    if timing is None:
        return ""
    return (
        '<section id="research-timing"><h3>Research timing</h3>'
        + _facts(
            {
                "Declared interval and research cutoff": timing["declared"],
                "Evaluated formations": timing["evaluated"],
                "Input observation range": timing["input"],
                "Source availability policy (not provider arrival)": timing["availability"],
                "Outcome clock": timing["outcome"],
                "Selected formation events": timing["selected_event"],
                "Training and maturity": timing["training"],
                "Portfolio rebalance clock": timing.get("rebalance"),
            }
        )
        + "".join("<p>" + _text(v) + "</p>" for v in timing["notices"])
        + temporal_scope_section(timing.get("temporal_scope"))
        + "</section>"
    )


def temporal_scope_section(scope: dict[str, Any] | None) -> str:
    """Render what the window can claim about time: its statements, then the marks (V347)."""
    if not scope or scope.get("status") != "RECORDED":
        return ""
    marks = {
        key: scope.get(key)
        for key in (
            "t0_session",
            "initial_cohort_size",
            "window_before_t0",
            "data_before_t0",
            "universe_basis",
            "survivorship_bias",
            "sector_treatment",
            "price_basis",
        )
    }
    return (
        '<h4 id="research-temporal-scope">Time and survivorship</h4>'
        + "".join("<p>" + _text(v) + "</p>" for v in scope.get("statements", ()))
        + _facts({"Recorded marks": marks})
    )


def _next_steps(body: dict[str, Any]) -> str:
    requests = body.get("next_requests", {})
    items = "".join(
        f"<li>{_text(name)}: {_text(request.get('operation'))}</li>"
        for name, request in requests.items()
        if isinstance(request, dict)
    )
    return (
        '<section id="research-next-steps"><h2>Continue from this result</h2>'
        "<p>Choose an offered action explicitly through the product. Each request is "
        "revalidated; a draft is not a PLAN and a PLAN is not execution permission. "
        "Companion JSON carries exact references and declarations.</p>"
        + ("<ul>" + items + "</ul>" if items else "<p>No next request recorded.</p>")
        + "<p>Recorded limitations: "
        + _text(body.get("limitations"))
        + "</p></section>"
    )


def render_experiment_report(body: dict[str, Any]) -> str:
    """Render a published experiment or its current Task status as HTML.

    Args:
        body: The owner's experiment answer.

    Returns:
        A complete HTML document for the answer.
    """
    if body.get("status") not in {"EXPERIMENT_PUBLISHED", "REUSED_EXACT"}:
        return (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>Research Task status</title><h1>Research Task status</h1>"
            "<p>No published numerical result is supplied for this Task.</p>"
            + _facts(
                {
                    "Status": body.get("status"),
                    "Task": body.get("task_id"),
                    "Failure": body.get("failure_code"),
                }
            )
            + _next_steps(body)
            + "</html>"
        )
    if body.get("program", {}).get("kind") == "risk.covariance-development":
        return _risk_report(body)
    if body.get("program", {}).get("kind") == "portfolio.policy-development":
        return _portfolio_report(body)
    if body.get("program", {}).get("kind") == "alpha.model-development":
        return _alpha_report(body)
    return _factor_report(body)


def risk_report_section(body: dict[str, Any], projection: dict[str, Any] | None = None) -> str:
    """Diagnostic facts, not a risk forecast attributed to a Portfolio allocation."""
    surface, diagnostic = body["risk_surface"], body["result"]
    fields = (
        "formation_session",
        "next_session",
        "shrinkage",
        "condition_number",
        "annualized_volatility_median",
        "equal_weight_predicted_variance",
        "equal_weight_realized_squared_return",
    )
    evaluations = projection["evaluations"] if projection is not None else diagnostic["evaluations"]
    facts = {
        "Risk method": surface["capability_handle"],
        "Recipe": body["document"]["risk"]["estimator"]["parameters"],
        "Input binding": body.get("input_binding_hash"),
        "Program authority": body["program"]["authority_hash"],
        "Cutoff": body["document"]["experiment"]["sessions"]["as_of"],
        "Published formations": len(diagnostic["evaluations"]),
    }
    window_notice = ""
    if projection is not None:
        window = projection["window"]
        facts["Source published formations"] = facts.pop("Published formations")
        facts["Displayed formations"] = len(evaluations)
        window_notice = (
            "<p>Explicit post-observed Portfolio-window reference. These diagnostics are not "
            "claimed to have been available at the Portfolio decision cutoff, and cannot "
            "be used to change allocation or claim prospective validation. The original "
            "Risk surface and diagnostics retain their full source identities.</p>"
        )
        facts["Source Risk cutoff"] = window["source_as_of"]
        facts["Portfolio cutoff"] = window["portfolio_as_of"]
    scope_surfaces = surface.get("scope_surfaces") or (surface,)
    scope_facts = (
        [
            (scope["first_session"], scope["last_session"], len(scope["ordered_listing_ids"]))
            for scope in projection["assessed_listing_scopes"]
        ]
        if projection is not None
        else [
            (
                scope["formation_sessions"][0],
                scope["formation_sessions"][-1],
                len(scope["ordered_listing_ids"]),
            )
            for scope in scope_surfaces
        ]
    )
    scope_rows = "".join(
        "<tr><td>"
        + _text(start)
        + "</td><td>"
        + _text(end)
        + "</td><td>"
        + _text(count)
        + "</td></tr>"
        for start, end, count in scope_facts
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{_text(row.get(k))}</td>" for k in fields) + "</tr>"
        for row in evaluations
    )
    return (
        '<section id="risk-diagnostics"><h2>Risk development diagnostics</h2>'
        "<p>Development-only evidence on the selected frozen input, without a full historical "
        "point-in-time membership claim. Covariance uses "
        "one-session log returns. Equal-weight and sector-balanced diagnostics describe "
        "reference books, not the risk of a user's different holdings. No allocation changes "
        "or current Risk admission.</p>"
        + window_notice
        + "<dl>"
        + "".join(f"<dt>{_text(k)}</dt><dd>{_text(v)}</dd>" for k, v in facts.items())
        + "</dl><h3>Dated estimation scopes</h3><p>Each scope has its own matrix axis. "
        "Stability comparisons restart when that axis changes.</p><table><thead><tr>"
        "<th>First formation</th><th>Last formation</th><th>Assets</th></tr></thead><tbody>"
        + scope_rows
        + '</tbody></table><div class="table-scroll"><table><thead><tr>'
        + "".join(f"<th>{html.escape(k.replace('_', ' '))}</th>" for k in fields)
        + "</tr></thead><tbody>"
        + rows
        + "</tbody></table></div></section>"
    )


def _risk_report(body: dict[str, Any]) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Risk development diagnostics</title><style>"
        "body{font:16px system-ui;margin:1rem;max-width:1100px}"
        "dl,p{overflow-wrap:anywhere}.table-scroll{overflow-x:auto}"
        "td,th{padding:.4rem;border:1px solid #bbb}table{border-collapse:collapse}</style>"
        + "<h1>Risk development diagnostics</h1>"
        + _study_context(body)
        + risk_report_section(body)
        + _next_steps(body)
        + "</html>"
    )


def render_linked_risk_report(snapshot: dict[str, Any]) -> str:
    """Append an explicitly selected reference; never change the original book."""
    link = snapshot["link"]
    notice = (
        "<section><h1>Linked Risk report</h1><p>Report reference only. The original "
        "Portfolio below did not consume this Risk model for sizing. Its economics "
        "and original receipt are unchanged.</p><p>Diagnostic coverage: "
        + html.escape(str(link["risk_formation_count"]))
        + " of "
        + html.escape(str(link["portfolio_formation_count"]))
        + " Portfolio sessions. "
        "This is not a full-lifecycle Risk assessment.</p><p>Reference: "
        + html.escape(link["link_hash"])
        + "</p></section>"
    )
    if link.get("assessed_listing_scopes"):
        notice += (
            "<p>The Risk reference assesses only its dated listing scopes shown below, "
            "not every historical Portfolio name or retained holding. Names outside those "
            "scopes have no covariance assessment in this reference.</p>"
        )
    return _portfolio_report(snapshot["portfolio"]).replace(
        "</html>",
        notice
        + risk_report_section(snapshot["risk"], snapshot.get("report_projection"))
        + "</html>",
        1,
    )


def _portfolio_report(body: dict[str, Any]) -> str:
    position = body["position"]
    source = body["portfolio_source"]
    facts = {
        "Source Alpha experiment": source["alpha_task_id"],
        "Candidate": source["candidate_id"],
        "Target meaning": source["target_recipe_id"],
        "Portfolio policy": body["document"]["portfolio"],
        "Selected holdings session": position["session"],
        "Cash": position["cash"],
        "Score sessions": body["execution_preview"]["score_session_count"],
        "Hold-only sessions": body["execution_preview"]["hold_session_count"],
        "Source universe names": body.get("data_quality", {}).get("source_listing_count"),
        "Effective Portfolio names": body.get("data_quality", {}).get("effective_listing_count"),
        "Data-quality policy": body.get("data_quality", {}).get("policy"),
        **(
            {"Data-quality fallback": body["data_quality"]["notice"]}
            if body.get("data_quality", {}).get("notice")
            else {}
        ),
        **body["result"],
    }
    holdings = "".join(
        "<tr>" + "".join(f"<td>{_text(v)}</td>" for v in (listing, target, weight)) + "</tr>"
        for listing, target, weight in zip(
            body.get("listing_labels") or source["ordered_listing_ids"],
            position["targets"],
            position["weights"],
            strict=True,
        )
        if target != 0 or weight != 0
    )
    series = "".join(
        "<tr>"
        + "".join(
            f"<td>{_text(row[k])}</td>"
            for k in (
                "session",
                "decision_mode",
                "gross_simple_return",
                "net_simple_return",
                "benchmark_simple_return",
                "one_way_turnover",
                "cost_fraction",
            )
        )
        + "</tr>"
        for row in body["series"]
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Portfolio development replay</title><style>"
        "body{font:16px system-ui;margin:1rem;max-width:1100px}"
        "dl,p{overflow-wrap:anywhere}.table-scroll{overflow-x:auto}"
        "td,th{padding:.4rem;border:1px solid #bbb}table{border-collapse:collapse}</style>"
        "<h1>Portfolio development replay</h1><p>Observed development evidence, "
        "not independent validation or a live recommendation. No Risk model, covariance forecast, "
        "model decomposition or strategy activation. Undefined ratios remain unavailable.</p>"
        "<p>Returns and weights are fractions; the eligible-universe equal-weight benchmark "
        "is a descriptive reference, not an investable benchmark.</p>"
        + _study_context(body)
        + "<dl>"
        + "".join(f"<dt>{_text(k)}</dt><dd>{_text(v)}</dd>" for k, v in facts.items())
        + "</dl>"
        '<div class="table-scroll"><table><caption>Selected session holdings</caption>'
        "<tr><th>Listing</th><th>Target</th><th>Executed</th></tr>" + holdings + "</table></div>"
        '<div class="table-scroll"><table><caption>Continuous economic calendar, '
        "including embargo HOLD sessions</caption>"
        "<tr><th>Session</th><th>Mode</th><th>Gross</th><th>Net</th><th>Benchmark</th>"
        "<th>One-way turnover</th><th>Cost fraction</th></tr>"
        + series
        + "</table></div><p>"
        + _text("; ".join(body["limitations"]))
        + "</p>"
        + _next_steps(body)
        + "</html>"
    )


def _factor_report(body: dict[str, Any]) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Factor research evidence</title><style>"
        "body{font:16px system-ui;margin:2rem;max-width:1100px}table{border-collapse:collapse}"
        "td,th{padding:.5rem;border:1px solid #bbb;text-align:left}dl{overflow-wrap:anywhere}"
        ".table-scroll{overflow-x:auto}p{line-height:1.5}</style>"
        + factor_report_section(body)
        + "</html>"
    )


def factor_report_section(body: dict[str, Any]) -> str:
    """Render the selected immutable facts; JSON remains the exact machine view."""
    program = body.get("program", {})
    document = body.get("document", {})
    selected = document.get("factor", {}).get("factor_ids", [])
    report = body.get("result", {}).get("evidence_report", {})
    quality = body.get("result", {}).get("target_quality", {})
    quality_content = (
        "<section><h2>Target availability</h2>"
        + _facts(
            {
                "Listings": quality.get("listing_count"),
                "Target rows": quality.get("row_count"),
                "Valid targets": quality.get("valid_target_count"),
                "Typed missing targets": quality.get("typed_missing_count"),
                "Missing reasons": quality.get("missing_reasons"),
                "Outcome snapshot": quality.get("source_outcome_snapshot_hash"),
            }
        )
        + "<p>Missing execution outcomes are not zero returns. These are the "
        "target owner's counts, not a new source-quality clearance.</p></section>"
    )
    sessions = body.get("evidence", {}).get("formation_sessions", [])
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{_text(item.get(key))}</td>"
            for key in (
                "factor_id",
                "classification",
                "mean_oriented_rank_ic",
                "rank_ic_by_q_value",
                "validation_pair_coverage_mean",
                "validation_rank_turnover_mean",
            )
        )
        + f"<td>{_text('; '.join(item.get('reason_codes', [])))}</td></tr>"
        for item in report.get("items", [])
        if item.get("factor_id") in selected
    )
    return (
        "<h1>Factor research evidence</h1>"
        "<p>Development evidence only. No Foundation admission, strategy activation"
        " or timely advice.</p>"
        + _study_context(body)
        + f"<dl><dt>Status</dt><dd>{_text(body.get('status'))}</dd>"
        f"<dt>Method</dt><dd>{_text(program.get('kind'))}</dd>"
        f"<dt>Program</dt><dd>{_text(program.get('program_hash'))}</dd>"
        f"<dt>Full statistical context</dt><dd>{_text(report.get('hypothesis_count'))} factors</dd>"
        f"<dt>Policy-derived development window</dt><dd>{_text(sessions[0] if sessions else None)}"
        f" — {_text(sessions[-1] if sessions else None)}</dd></dl>"
        "<p>The selected factors scope this view. The statistical window and multiple-testing"
        " denominator remain those of the admitted policy and full context.</p>"
        '<div class="table-scroll"><table><caption>Selected Factor evidence, '
        "as reported by its owner</caption>"
        "<thead><tr><th>Factor</th><th>Classification</th><th>Oriented rank IC</th>"
        "<th>BY q value</th><th>Pair coverage</th><th>Rank turnover</th>"
        "<th>Reasons</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        + quality_content
        + "<p>Download the companion JSON for exact receipts,"
        " inputs, policies and full diagnostics.</p>" + _next_steps(body)
    )


def _alpha_report(body: dict[str, Any]) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Alpha development evidence</title><style>"
        "body{font:16px system-ui;margin:1rem;max-width:1100px}"
        "dl,p{overflow-wrap:anywhere}.table-scroll{overflow-x:auto}"
        "td,th{padding:.5rem;border:1px solid #bbb}table{border-collapse:collapse}"
        "</style>" + alpha_report_section(body) + "</html>"
    )


def alpha_report_section(body: dict[str, Any]) -> str:
    """Render the Alpha result or lifecycle reconstruction section.

    Args:
        body: The owner's Alpha result answer.

    Returns:
        HTML for the report section.
    """
    if "alpha_qualification" in body:
        # A family's qualification concludes over every attempted candidate; it holds no
        # development result, which the section below reads (V494).
        qualification = body["alpha_qualification"]
        return (
            "<h1>Alpha family qualification</h1>"
            "<p>Qualified over the whole attempted family, the sealed holdout unread; not a "
            "current strategy activation.</p>"
            + _study_context(body)
            + _facts(
                {
                    "Disposition": qualification.get("disposition"),
                    "Attempted candidates": len(qualification.get("attempted_candidate_ids", ())),
                    "Selected candidates": qualification.get("selected_candidate_ids"),
                    "Currently qualified": qualification.get("current_qualified_ids"),
                    "Fits": qualification.get("fit_call_count"),
                    "Predictions": qualification.get("predict_call_count"),
                    "Qualification": qualification.get("qualification_hash"),
                }
            )
            + _next_steps(body)
        )
    if "lifecycle_research" in body:
        receipt, preview = body["lifecycle_research"], body["execution_preview"]
        return (
            "<h1>Model lifecycle research</h1>"
            "<p>Local reconstruction on this input, not the original historical package "
            "or a current strategy activation. Scoring is not Portfolio execution.</p>"
            + _study_context(body)
            + _facts(
                {
                    "Component": preview.get("component_id"),
                    "Model features": preview.get("ordered_feature_ids"),
                    "Lifecycle": receipt["lifecycle"],
                    "Required vintages": preview.get("required_vintages"),
                    "Original fits": receipt.get("fit_call_count"),
                    "Original score predictions": receipt.get("prediction_call_count"),
                    "Scored formation count": len(receipt["formation_sessions"]),
                    "Receipt": receipt["content_hash"],
                }
            )
            + _next_steps(body)
        )
    declaration = body["document"]["alpha"]
    preview = body["execution_preview"]
    result = body["result"]
    facts = {
        "Status": body["status"],
        "Program": body["program"]["program_hash"],
        "Source Factor experiment": body["alpha_source"]["factor_task_id"],
        "Factor decision": body["alpha_source"]["curation_receipt_hash"],
        "Research Foundation": body["alpha_source"].get(
            "foundation_admission_hash", "Development input only"
        ),
        "Target": declaration["target_recipe_id"],
        "Model": declaration["model_parameters"],
        "Features": declaration["ordered_feature_ids"],
        "Policy-derived validation start": preview["statistical_start"],
        "Policy-derived validation end": preview["statistical_end"],
        "Recorded fits in completing execution": result["fit_call_count"],
        "Recorded predictions": result["predict_call_count"],
        "Recorded metric calls": result["metric_call_count"],
    }
    metrics = ("mae", "mse", "zero_relative_oos_r2", "rank_ic", "gross_decile_spread")
    rows = "".join(
        f"<tr><td>{_text(fold['fold_index'])}</td>"
        + "".join(f"<td>{_text((fold.get('metrics') or {}).get(key))}</td>" for key in metrics)
        + "</tr>"
        for fold in body["fold_results"]
    )
    summary = "".join(
        "<tr>"
        + "".join(
            f"<td>{_text(candidate.get(key))}</td>"
            for key in (
                "candidate_id",
                "pooled_oos_r2",
                "mean_rank_ic",
                "mean_gross_decile_spread",
                "fold_coverage_mean",
            )
        )
        + "</tr>"
        for candidate in result["candidates"]
    )
    return (
        "<h1>Alpha development evidence</h1>"
        "<p>Development measurement only. A named research Foundation is not strategy activation, "
        "independent scientific validation or timely advice. No winner is selected.</p>"
        + _study_context(body)
        + "<dl>"
        + "".join(f"<dt>{_text(k)}</dt><dd>{_text(v)}</dd>" for k, v in facts.items())
        + "</dl><p>Counts describe the completing execution; verified children may come "
        "from earlier interrupted work. An exact replay performs no new numerical calls.</p>"
        '<div class="table-scroll"><table><caption>Recorded candidate summary</caption>'
        "<tr><th>Candidate</th><th>OOS R²</th><th>Rank IC</th>"
        "<th>Gross spread</th><th>Coverage</th></tr>"
        + summary
        + '</table></div><div class="table-scroll"><table><caption>Verified folds</caption>'
        "<tr><th>Fold</th><th>MAE</th><th>MSE</th><th>OOS R²</th>"
        "<th>Rank IC</th><th>Gross spread</th></tr>"
        + rows
        + "</table></div><p>Companion JSON contains exact receipts, child identities, "
        "full metrics and fold windows; YAML preserves the authored declaration.</p>"
        + _next_steps(body)
    )


def render_research_delivery(snapshot: dict[str, Any]) -> str:
    """Compose existing verified content; commentary stays escaped and non-authoritative."""
    from alphalattice.interface.local_application.evidence_cro import render_review_export

    sections = snapshot["sections"]
    primary = _portfolio_report(sections["portfolio"]["value"])
    notice = (
        '<section id="delivery-context"><h2>Research delivery</h2><p>'
        + _text(snapshot["claim"])
        + "</p>"
        + _facts(
            {
                "Question for this delivery": snapshot["question"],
                "Question provenance": snapshot["question_status"],
                **snapshot["input"],
            }
        )
        + '<nav><a href="#delivery-comparison">Comparison</a> · '
        '<a href="#delivery-risk">Risk reference</a> · '
        '<a href="#delivery-evidence-state">Evidence/CRO state</a> · '
        '<a href="#delivery-commentary">Attributed commentary</a></nav></section>'
    )
    parts = []
    for name, renderer in (("factor", factor_report_section), ("alpha", alpha_report_section)):
        content = renderer(sections[name]["value"])
        for anchor in ("study-context", "research-next-steps"):
            content = content.replace(f'id="{anchor}"', f'id="{name}-{anchor}"')
        content = content.replace("<h1>", "<h2>", 1).replace("</h1>", "</h2>", 1)
        parts.append(
            f'<details id="delivery-{name}"><summary>{name.title()} evidence</summary>'
            + content
            + "</details>"
        )
    foundation = sections["foundation"]
    parts.append(
        '<section id="delivery-foundation"><h2>Research Foundation</h2>'
        + _facts(foundation)
        + "</section>"
    )
    comparison = sections["comparison"]
    parts.append('<section id="delivery-comparison"><h2>Declared comparison</h2>')
    if comparison["status"] == "PRESENT":
        value = comparison["value"]
        parts.append(
            "<p>"
            + _text(value["claim"])
            + "</p>"
            + _facts(
                {
                    "Left Task": value["left"]["task_id"],
                    "Right Task": value["right"]["task_id"],
                }
            )
        )
        rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{_text(v)}</td>"
                for v in (
                    group["dimension"],
                    metric["label"],
                    metric["unit"],
                    metric["left"],
                    metric["right"],
                )
            )
            + "</tr>"
            for group in value["dimensions"]
            for metric in group["metrics"]
        )
        parts.append(
            '<div class="table-scroll"><table><tr><th>Dimension</th><th>Metric</th>'
            "<th>Unit</th><th>Left</th><th>Right</th></tr>" + rows + "</table></div>"
        )
    else:
        parts.append(_facts(comparison))
    parts.append('</section><section id="delivery-risk"><h2>Risk report reference</h2>')
    risk = sections["risk"]
    if risk["status"] == "PRESENT":
        value = risk["value"]
        parts.append(
            "<p>This linked Risk model did not change Portfolio sizing. "
            "Its coverage may be shorter than the Portfolio study.</p>"
            + _facts(
                {"Selected holdings date covered by Risk": risk.get("selected_session_covered")}
            )
            + _facts(value["link"])
            + risk_report_section(value["risk"], value.get("report_projection"))
        )
    else:
        parts.append(_facts(risk))
    review = sections["evidence_cro"]
    parts.append(
        '</section><section id="delivery-evidence-state"><h2>Evidence/CRO state</h2>'
        + _facts({k: v for k, v in review.items() if k != "value"})
        + "</section>"
    )
    parts.append(
        '<section id="delivery-commentary"><h2>Attributed commentary, not authority</h2>'
        + _facts(snapshot["commentary_provenance"])
    )
    for item in snapshot["commentary"]:
        parts.append(
            "<article><h3>"
            + _text(item["attribution"])
            + '</h3><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'
            + _text(item["text"])
            + "</pre></article>"
        )
    if not snapshot["commentary"]:
        parts.append("<p>No commentary supplied; no historical intent inferred.</p>")
    parts.append("</section>")
    result = primary.replace("<h1>", notice + "<h1>", 1).replace(
        "</html>", "".join(parts) + "</html>", 1
    )
    if "value" in review:
        result = render_review_export(review["value"], result)
    return result


def render_handoff_report(body: dict[str, Any]) -> str:
    """A readable decision and preparation receipt, never a performance claim."""
    decision = body["curation"]
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{_text(choice[key])}</td>" for key in ("factor_id", "role", "rationale"))
        + "</tr>"
        for choice in decision["submission"]["proposal"]["choices"]
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Factor development handoff</title><style>"
        "body{font:16px system-ui;margin:1rem;max-width:1100px}"
        "td,th{padding:.5rem;border:1px solid #aaa}table{border-collapse:collapse}"
        "div{overflow-x:auto}"
        "p,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>"
        "<h1>Factor development handoff</h1><p>Development research only. "
        "No Foundation publication, "
        "training, strategy activation or timely advice.</p>"
        f"<p>Status: {_text(body['status'])}</p><p>Decision: {_text(decision['receipt_hash'])}</p>"
        f"<p>Input binding: {_text(body['input_binding_hash'])}</p>"
        "<div><table><caption>Explicit researcher selection</caption><thead><tr>"
        f"<th>Factor</th><th>Role</th><th>Reason</th></tr></thead><tbody>{rows}</tbody></table></div>"
        f"<p>{_text('; '.join(body['limitations']))}</p><h2>Exact preparation</h2>"
        f"<pre>{_text(json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2))}</pre></html>"
    )
