"""The public path's single self-contained static HTML renderer.

Everything on the page is a verified fact from a typed owner. No metric,
compounding, ratio or unit conversion happens here -- if a number is not already
on the report, the economic ledger, the benchmark comparison or the coverage, it
is not shown. That rule is what keeps the renderer from quietly becoming a second
formula owner.

No JavaScript, no font host, no CDN, no image host: the page is one file that
opens with the network off and prints without a layout pass. The only graphic is
an inline SVG drawn from the report's own rows.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from alphalattice.investment.portfolio_strategy_lab.application.contracts import (
    PortfolioBenchmarkComparison,
    PortfolioDeclaredPathReport,
    PortfolioEconomicLedger,
    PortfolioResearchSpec,
    PortfolioSupportCoverage,
    PortfolioWindowEndBook,
)
from alphalattice.investment.portfolio_strategy_lab.application.controls import (
    EVIDENCE_COST_LADDER_BPS_PER_SIDE,
    INSTALLED_PUBLIC_CONTROL_CATALOG,
)
from alphalattice.investment.portfolio_strategy_lab.application.strategy_package import (
    FrozenStrategyPackage,
    StrategyDisclosure,
)


@dataclass(frozen=True)
class PortfolioReportContext:
    """Source-bound descriptive facts, not another numerical report or authority."""

    program_hash: str
    book_policy: Mapping[str, object]
    listing_labels: Mapping[str, str]


if TYPE_CHECKING:
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        PortfolioDecisionCheckpoint,
        PortfolioUpdatePublication,
    )


def render_decision_update(
    checkpoint: PortfolioDecisionCheckpoint, history: tuple[PortfolioUpdatePublication, ...]
) -> str:
    """Render already-computed QA values, independently of models or network."""
    from alphalattice.investment.portfolio_strategy_lab.application.decision_updates import (
        portfolio_update_positions,
    )

    current = history[-1]
    proposal = current.pending_proposal
    facts = portfolio_update_positions(current, history)
    rows = []
    weights = facts.weights
    changes = facts.changes
    previous = facts.preceding or weights
    for index, (listing, label, weight) in enumerate(
        zip(checkpoint.ordered_listing_ids, checkpoint.listing_labels, weights, strict=True)
    ):
        delta = (
            "Not available in this historical artifact" if changes is None else str(changes[index])
        )
        if weight > 0 or previous[index] > 0:
            rows.append(
                f"<tr><td>{html.escape(label)}</td><td>{html.escape(listing)}</td>"
                f"<td>{weight:.8f}</td><td>{delta}</td></tr>"
            )
    events = []
    seen_proposals: set[str] = set()
    for publication in history:
        if (
            publication.pending_proposal is not None
            and publication.pending_proposal.content_hash not in seen_proposals
        ):
            p = publication.pending_proposal
            seen_proposals.add(p.content_hash)
            events.append(
                f"<tr><td>{p.schedule.formation_session}</td><td>PROPOSAL_PUBLISHED</td>"
                f"<td>{p.schedule.entry_session}</td><td>Pending</td><td>Pending</td>"
                "<td>Pending</td><td>Pending</td></tr>"
            )
        for event in publication.events:
            returns = "Pending" if event.gross_return is None else str(event.gross_return)
            net5 = "Pending" if event.net_return_5bps is None else str(event.net_return_5bps)
            net10 = "Pending" if event.net_return_10bps is None else str(event.net_return_10bps)
            events.append(
                f"<tr><td>{event.formation_session}</td><td>{event.phase}</td>"
                f"<td>{event.entry.schedule.entry_session}</td><td>{event.turnover}</td>"
                f"<td>{returns}</td><td>{net5}</td><td>{net10}</td></tr>"
            )
    label = (
        "CLOSE_MARKED_ESTIMATE_NOT_EXECUTION_TARGET"
        if proposal
        else "OBSERVED_RESEARCH_ENTRY_WEIGHTS"
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Portfolio decision and settlement QA</title><style>{_STYLE}</style><main>"
        "<h1>Portfolio decision and settlement QA</h1>"
        f"<p>{html.escape(checkpoint.package.strategy_id)} · {current.claim}</p>"
        f"<p>Observed through {current.observed_through}; "
        f"published {current.published_at.isoformat()}</p>"
        f"<p>{label}. Weights are portfolio fractions, not account holdings or orders.</p>"
        "<p>Final weights evaluate the original sealed inputs against observed entry prices. "
        "Daily observations do not imply real-time execution. "
        "Risk and CRO: not evaluated for this proposal. "
        "Fixed 5/10 bps per-side cost sensitivity; "
        "PIT, delisting and capacity authority remain incomplete.</p>"
        + (
            "<p>Source revision: "
            + str(current.source_revision.changed_bar_count)
            + " prior bars differ. Continue from as-issued holdings; prior publications "
            "are unchanged. Revised replay was not performed or adopted. "
            "Unverified split-basis changes require review.</p>"
            if current.source_revision is not None
            else ""
        )
        + '<div class="scroll"><table><thead><tr><th>Name</th><th>Listing</th>'
        "<th>Weight (fraction)</th><th>Change from close (fraction)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div><h2>As-issued history and observed settlement</h2>"
        '<div class="scroll"><table><thead><tr><th>Decision</th><th>Phase</th>'
        "<th>Entry</th><th>One-way turnover</th><th>Gross return</th>"
        "<th>Net, 5 bps/side</th><th>Net, 10 bps/side</th></tr></thead><tbody>"
        + "".join(events)
        + "</tbody></table></div></main></html>"
    )


_STYLE = """
:root{--ink:#16202b;--muted:#5b6875;--rule:#d6dde6;--ground:#ffffff;--panel:#f6f8fa;
--accent:#1f5f8b;--warn:#8a5a00}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;padding:32px 20px 64px;
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:60rem;margin:0 auto}
html,body{overflow-x:hidden}
h1{font-size:1.65rem;margin:0 0 .25rem}
h2{font-size:1.1rem;margin:2.25rem 0 .6rem;padding-bottom:.3rem;border-bottom:1px solid var(--rule)}
p.lede{color:var(--muted);margin:0 0 1.5rem}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;min-width:22rem;font-variant-numeric:tabular-nums}
caption{text-align:left;color:var(--muted);padding-bottom:.4rem}
th,td{padding:7px 10px;border-bottom:1px solid var(--rule);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}
thead th{border-bottom:2px solid var(--rule);font-weight:600}
dl.facts{display:grid;grid-template-columns:minmax(11rem,auto) 1fr;gap:.35rem 1.25rem;margin:0}
dl.facts dt{color:var(--muted)}
dl.facts dd{margin:0;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 16px}
.empty{color:var(--muted);font-style:italic;padding:10px 0}
.tag{display:inline-block;border:1px solid var(--rule);border-radius:12px;
padding:2px 10px;font-size:12px;color:var(--muted);max-width:100%;overflow-wrap:anywhere}
.warn{color:var(--warn)}
figure.spark{margin:.6rem 0 0}
/* Short labelled cells, not hashes: breaking them mid-word to fit a phone
   makes the book unreadable. Let the table be its own width and scroll. */
table.book th,table.book td{white-space:nowrap;overflow-wrap:normal}
ul.plain{margin:.3rem 0 0;padding-left:1.1rem}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
.skip{position:absolute;left:-9999px}
.skip:focus{position:static;display:inline-block;margin-bottom:1rem}
@media (max-width:640px){
body{padding:18px 12px 48px}
dl.facts{grid-template-columns:1fr;gap:.15rem}
dl.facts dt{margin-top:.55rem}
h1{font-size:1.35rem}}
@media print{
body{padding:0;font-size:11pt}
h2{break-after:avoid}
table,figure{break-inside:avoid}
.panel{background:none}
a[href]::after{content:""}}
"""


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _rows(cells: list[tuple[str, str]]) -> str:
    return "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>"
        for name, value in cells
    )


def _table(
    caption: str, headers: tuple[str, ...], body: str, *, empty: str, css_class: str = ""
) -> str:
    if not body:
        return f'<div class="panel"><p class="empty">{html.escape(empty)}</p></div>'
    head = "".join(f"<th scope='col'>{html.escape(value)}</th>" for value in headers)
    opening = f'<table class="{css_class}">' if css_class else "<table>"
    return (
        f'<div class="scroll">{opening}<caption>{html.escape(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _book_rows(book: PortfolioWindowEndBook, labels: Mapping[str, str] | None = None) -> str:
    """One row per name in the sealed book. Every value is escaped.

    A listing id is a universe symbol, not markup, and this table is the one
    place a report renders identifiers that came from outside it -- so each cell
    goes through `html.escape` even though today's universe is plain tickers.
    """

    return "".join(
        "<tr><td>"
        + (
            f"{html.escape(labels[position.listing_id])}<br>"
            if labels and position.listing_id in labels
            else ""
        )
        + f"{html.escape(position.listing_id)}</td>"
        f"<td>{format_book_weight(position.weight)}</td>"
        f"<td>{format_book_weight(position.preceding_weight)}</td>"
        f"<td>{format_book_change(position.weight_change)}</td>"
        f"<td>{html.escape(position.disposition)}</td></tr>"
        for position in book.positions
    )


def _boundary_label(book: PortfolioWindowEndBook) -> str:
    """What the change is measured against, in words a reader can act on.

    Three different things, and the difference matters more than it looks. A
    continuation's first formation has a real predecessor whose names it carried
    in; calling that an empty opening would tell the reader the whole book was
    bought that morning, and would hide every reduction and every exit.
    """

    if book.change_boundary == "SEALED_CONTINUATION_BOUNDARY":
        session = (
            ""
            if book.preceding_formation_session is None
            else (book.preceding_formation_session.isoformat())
        )
        return (
            f"{html.escape(session)} &mdash; the sealed boundary this path "
            "continued from, not an empty opening"
        )
    if book.change_boundary == "FLAT_PATH_OPENING":
        return "an empty book &mdash; this path&rsquo;s first formation opened flat"
    session = (
        ""
        if book.preceding_formation_session is None
        else (book.preceding_formation_session.isoformat())
    )
    return html.escape(session)


def format_book_weight(value: float) -> str:
    """A share of the book, to a thousandth of a percent.

    Two decimals would print a real position of four basis points as `0.00%`,
    and a holdings table that renders a held name as nothing is worse than one
    that is hard to skim.

    Public because the sealed page is not the only consumer: the local product
    projects the same book as JSON, and a weight written one way here and
    another way there would be two presentations of one number.
    """
    return f"{value * 100:.3f}%"


def format_book_change(value: float) -> str:
    """A change, signed, in basis points of the book.

    Percent at any readable precision collapses one session's drift to zero.
    Basis points are the unit the cost contract on this same page already
    teaches, and they carry the resolution a change actually has.
    """
    return f"{value * 10_000:+.2f}"


def _sparkline(rows: tuple[tuple[str, float], ...]) -> str:
    """One inline SVG drawn from the report's own rows. No library, no request."""

    if len(rows) < 2:
        return ""
    values = [value for _label, value in rows]
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    width, height = 640.0, 120.0
    step = width / (len(values) - 1)
    points = " ".join(
        f"{index * step:.2f},{height - (value - low) / span * (height - 8) - 4:.2f}"
        for index, value in enumerate(values)
    )
    label = (
        f"{html.escape(rows[0][0])} to {html.escape(rows[-1][0])}, "
        f"low {_pct(low)}, high {_pct(high)}"
    )
    return (
        f'<figure class="spark"><svg viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="100%" height="120" role="img" aria-label="{label}" '
        'preserveAspectRatio="none">'
        f'<polyline fill="none" stroke="#1f5f8b" stroke-width="2" points="{points}"/>'
        f"</svg><figcaption class='tag'>{label}</figcaption></figure>"
    )


def _headline(spec: PortfolioResearchSpec) -> str:
    """What this page is, in the mode it was produced under.

    A replay describes a book that was selected and held historically. Calling it
    "today" would be a forward claim the path cannot support, and the wording is
    decided here rather than left to whoever writes the next template.
    """

    if spec.execution_mode == "DEVELOPMENT_REPLAY":
        return "Selected historical book and path"
    return "Declared Portfolio Path"


def _subject(spec: PortfolioResearchSpec) -> str:
    if spec.execution_mode == "DEVELOPMENT_REPLAY":
        return (
            f"{spec.execution_mode} \u00b7 {spec.weight_rule} \u00b7 "
            "a historical book and the changes it made over the selected window. "
            "No position here is held, intended or proposed."
        )
    return f"{spec.execution_mode} \u00b7 {spec.weight_rule}"


def _round_trip(economics: PortfolioEconomicLedger) -> str:
    """Both sides of the per-side rate, as the ledger already states it.

    The platform's one-way figure and the round trip are the same number here by
    construction, and both are carried so a reader is never asked to infer that.
    """

    return str(economics.platform_one_way_cost_bps)


_MERGE_NOTE: dict[str, str] = {
    "SINGLE_COMPONENT_BOOK": (
        "One component book carries the whole allocation; turnover, costs and "
        "returns are computed once on it."
    ),
    "POST_TRADE_LISTING_WEIGHTS_THEN_RECOMPUTE_ECONOMICS": (
        "Independent component books are formed first. Their post-trade listing "
        "weights are merged at the declared shares; turnover, costs and returns "
        "are then computed once on the merged book. This is not a score blend."
    ),
}
"""What each declared merge actually does, in the reader's words."""


def render_portfolio_research_html(
    *,
    spec: PortfolioResearchSpec,
    report: PortfolioDeclaredPathReport,
    economics: PortfolioEconomicLedger,
    comparison: PortfolioBenchmarkComparison,
    coverage: PortfolioSupportCoverage,
    disclosure: StrategyDisclosure | None = None,
    package: FrozenStrategyPackage | None = None,
    context: PortfolioReportContext | None = None,
) -> str:
    """Render verified facts only; no metric or selection arithmetic lives here."""
    schedule = report.schedule_guard
    if context is not None and context.program_hash != report.program_hash:
        raise ValueError("portfolio_report.context_program_mismatch")
    if package is not None and (
        disclosure is None
        or package.package_hash != disclosure.package_hash
        or package.strategy_id != disclosure.strategy_id
        or disclosure.program_hash != report.program_hash
        or disclosure.report_hash != report.report_hash
    ):
        raise ValueError("portfolio_report.package_disclosure_mismatch")
    frozen = package is not None and package.policy_binding == "PACKAGE_FROZEN"
    unresolved_policy = package is None and disclosure is not None
    configuration = (
        "package-frozen configuration"
        if frozen
        else "package policy details unavailable"
        if unresolved_policy
        else ("frozen default" if spec.is_default() else "bounded exploration")
    )
    control_note = (
        "This package fixes its book controls. A non-default package is not an "
        "exploratory alteration of the default strategy; reading it does not activate it."
        if frozen
        else "The package was recorded, but its policy details were not supplied to this "
        "renderer. No effective book policy is inferred from a generic control guard."
        if unresolved_policy
        else "The default control tuple and admitted control variations have separate "
        "configuration identities. A variation is never promoted to a default by this report."
    )
    window = report.window_guard
    beta_display = (
        "NOT_ESTIMABLE" if economics.benchmark_beta is None else f"{economics.benchmark_beta:.6f}"
    )
    clamp_note = (
        '<span class="tag">narrowed to the materialized path</span>'
        if window.clamped_to_materialized_path
        else ""
    )

    controls = _rows(
        [
            ("top_k per sleeve", str(spec.top_k)),
            ("tranches", str(spec.tranches)),
            ("weight_rule", spec.weight_rule),
            ("exit_rank", f"{spec.exit_rank} ({spec.exit_rank / spec.top_k:g}x top_k)"),
            ("cost (per side)", f"{economics.cost_bps_per_side} bps"),
            ("benchmark view", spec.secondary_benchmark_view),
            ("report unit", spec.report_unit),
            ("configuration", configuration),
        ]
    )
    ladder = _rows(
        [
            (
                f"{value} bps per side",
                f"{float(value) * 2:g} bps one-way turnover"
                + ("  <- selected" if value == str(economics.cost_bps_per_side) else ""),
            )
            for value in EVIDENCE_COST_LADDER_BPS_PER_SIDE
        ]
    )
    schedule_rows = _rows(
        [
            (
                f"formation {index}",
                "sleeves " + ", ".join(str(value) for value in due),
            )
            for index, due in enumerate(schedule.due_sleeve_cycle)
        ]
    )
    if frozen or unresolved_policy:
        policy = context.book_policy if context is not None else {}
        schedule_section = _table(
            "Effective package book policy",
            ("Policy fact", "Recorded value"),
            _rows([(key, str(value)) for key, value in policy.items() if key != "notice"]),
            empty="Detailed book-policy context was not recorded; the package policy governs.",
        ) + (
            '<p class="tag">A generic public-control schedule guard alone does not '
            "establish this package's effective policy. Recipe settings describe their declared "
            "modes; a sizing activation setting does not imply an unused sizing mode ran.</p>"
        )
    else:
        schedule_section = _table(
            f"{schedule.guard_family} at tranches={schedule.tranches}",
            ("Formation", "Due"),
            schedule_rows,
            empty="No schedule.",
        ) + (
            '<p class="tag">Schedule phase is not selectable '
            f"({html.escape(schedule.schedule_phase_refusal)}); the offset only relabels "
            "which sleeve trades first. Sleeve share: "
            f"{html.escape(schedule.sleeve_share_policy)}.</p>"
        )
    coverage_rows = "".join(
        "<tr><td>"
        + html.escape(owner.owner_id)
        + "</td><td>"
        + html.escape(owner.lane)
        + "</td><td>"
        + owner.first_session.isoformat()
        + "</td><td>"
        + owner.last_session.isoformat()
        + "</td><td>"
        + str(owner.session_count)
        + "</td></tr>"
        for owner in coverage.owners
    )
    risk = report.risk_facts[-1] if report.risk_facts else None
    risk_rows = ""
    if risk is not None:
        risk_rows = "".join(
            "<tr><td>"
            + html.escape(factor)
            + "</td><td>"
            + f"{exposure:.6f}"
            + "</td><td>"
            + f"{contribution:.8g}"
            + "</td></tr>"
            for factor, exposure, contribution in zip(
                risk.ordered_factor_ids,
                risk.industry_exposure,
                risk.factor_variance_contribution,
                strict=True,
            )
        )
    unit_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{_pct(value)}</td></tr>"
        for label, value in report.window_unit_rows
    )
    secondary = (
        f"{html.escape(comparison.secondary_benchmark_id or '')} "
        f"({html.escape(comparison.secondary_disposition)})"
        if comparison.secondary_benchmark_id
        else html.escape(comparison.secondary_disposition)
    )
    liquidity = (
        f"{report.window_end_median_holding_adv20_dollar_volume:.6g}"
        if report.window_end_median_holding_adv20_dollar_volume is not None
        else "NOT_AVAILABLE"
    )
    boundary = _boundary_label(report.window_end_book)
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in report.limitations)
    refusals = ", ".join(
        html.escape(value.refusal_code) for value in INSTALLED_PUBLIC_CONTROL_CATALOG.refusals
    )
    strategy_section = ""
    if disclosure is not None:
        component_rows = "".join(
            "<tr><td>"
            + html.escape(value.component_id)
            + "</td><td>"
            + html.escape(value.family_id)
            + "</td><td>"
            + f"{value.allocation_basis_points / 100:.0f}%"
            + "</td><td>"
            + html.escape(value.target_recipe)
            + "</td><td>"
            + html.escape(value.objective)
            + "</td></tr>"
            for value in disclosure.component_plan
        )
        claim_limits = "".join(
            f"<li>{html.escape(value)}</li>" for value in disclosure.claim_limits
        )
        provenance = "".join(f"<li>{html.escape(value)}</li>" for value in disclosure.provenance)
        # One section for every installed strategy. It reads the disclosure's
        # own fields rather than branching on which strategy produced it, which
        # is what lets a newly installed package be reported without a renderer
        # change.
        strategy_section = f"""
<h2 id="strategy">Frozen strategy package</h2>
<p class="tag">{html.escape(_MERGE_NOTE[disclosure.merge_semantics])}</p>
{
            _table(
                "Installed component disclosures",
                ("Component", "Family", "Allocation", "Target", "Objective"),
                component_rows,
                empty="No component disclosure.",
            )
        }
<div class="panel"><dl class="facts">
<dt>Strategy</dt><dd>{html.escape(disclosure.strategy_id)}</dd>
<dt>Package</dt><dd><code>{disclosure.package_hash}</code></dd>
<dt>Alpha recipe</dt><dd><code>{disclosure.alpha_recipe_hash}</code></dd>
<dt>Score source</dt><dd>{html.escape(disclosure.score_source_description)}
 ({html.escape(disclosure.score_source_mode)})</dd>
<dt>Score authority</dt><dd><code>{disclosure.score_source_authority_hash}</code></dd>
<dt>Merge</dt><dd>{html.escape(disclosure.merge_semantics)}</dd>
<dt>Risk role</dt><dd>{html.escape(disclosure.risk_disposition)}</dd>
<dt>Disclosure</dt><dd><code>{disclosure.disclosure_hash}</code></dd>
</dl></div>
<ul class="plain">{provenance}</ul>
<ul class="plain">{claim_limits}</ul>
"""
    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(_headline(spec))}</title>
<style>{_STYLE}</style>
<a class="skip" href="#portfolio">Skip to the Portfolio chapter</a>
<main>
<h1>{html.escape(_headline(spec))}</h1>
<p class="lede">{html.escape(_subject(spec))}</p>

<div class="panel" id="cost-contract">
<h2 class="masthead">Cost contract</h2>
<dl class="facts">
<dt>Per side</dt><dd>{html.escape(economics.cost_bps_per_side)} bps</dd>
<dt>Round trip</dt><dd>{html.escape(_round_trip(economics))} bps</dd>
<dt>Platform charging convention</dt><dd>{
        html.escape(economics.platform_one_way_cost_bps)
    } bps charged against <strong>one-way turnover</strong></dd>
</dl>
<p class="tag">All three are shown because a single unlabelled figure is the wrong
number for someone. A broker quotes per side; a round trip is both sides; the
engine charges its rate against one-way turnover, and one-way turnover is half
the absolute weight change.</p>
</div>

<h2 id="portfolio">Portfolio &mdash; the book and how it changed</h2>
<p class="tag">Every figure in this chapter is read from the sealed execution and
economic ledgers of this exact path. Nothing here was computed while rendering.</p>
<div class="panel"><dl class="facts">
<dt>Distinct names held at window end</dt><dd>{report.window_end_distinct_names}</dd>
<dt>Effective N at window end</dt><dd>{report.window_end_effective_n:.4f}</dd>
<dt>One-way turnover per trading session</dt>
<dd>{report.mean_one_way_turnover:.6f}</dd>
<dt>Aggregate-cap binding sessions</dt><dd>{report.aggregate_cap_binding_sessions}</dd>
<dt>Aggregate-cap affected names</dt><dd>{report.aggregate_cap_binding_names_total}</dd>
<dt>Median holding 20-session dollar volume</dt><dd>{liquidity}
<span class="tag">liquidity proxy, not a capacity estimate</span></dd>
<dt>Capacity</dt><dd class="warn">{html.escape(report.capacity_disposition)}</dd>
</dl></div>
<p class="tag">Distinct names is emergent from sleeve overlap; it is never
top_k &times; tranches. Turnover is per trading session, not annual.</p>
{strategy_section}

<h3>The book at {html.escape(report.window_end_book.formation_session.isoformat())}</h3>
<div class="panel"><dl class="facts">
<dt>Names held</dt><dd>{report.window_end_book.held_count}</dd>
<dt>Opened since the boundary</dt><dd>{report.window_end_book.opened_count}</dd>
<dt>Exited since the boundary</dt><dd>{report.window_end_book.exited_count}</dd>
<dt>Change measured against</dt><dd>{boundary}</dd>
<dt>Boundary kind</dt><dd>{html.escape(report.window_end_book.change_boundary)}</dd>
<dt>Total absolute weight change</dt>
<dd>{format_book_weight(report.window_end_book.absolute_weight_change_total)}</dd>
</dl></div>
{
        _table(
            "Every name held at the window end, and every name that left",
            ("Name", "Weight", "At boundary", "Change (bp)", "Disposition"),
            _book_rows(report.window_end_book, None if context is None else context.listing_labels),
            empty="This path held no position at the window end.",
            css_class="book",
        )
    }
<p class="tag">Weights are shares of the book. The change is between two
formation-end books, so it includes drift between them; it is not the traded
turnover above, which this path measures against the pre-trade book.</p>

<h2 id="performance">Performance &mdash; wealth and benchmark</h2>
<div class="panel"><dl class="facts">
<dt>Cumulative net wealth (selected window)</dt>
<dd>{report.window_cumulative_net_wealth:.6f}</dd>
<dt>Cumulative net wealth (full materialized path)</dt>
<dd>{economics.cumulative_net_wealth:.6f}</dd>
<dt>Primary benchmark</dt><dd>{html.escape(comparison.primary_benchmark)}</dd>
<dt>Benchmark view</dt><dd>{html.escape(comparison.secondary_benchmark_view)}</dd>
<dt>Secondary comparator</dt><dd>{secondary}</dd>
<dt>Beta to anchor</dt><dd>{beta_display} <span class="tag">{
        html.escape(economics.benchmark_beta_disposition)
    }</span></dd>
</dl></div>
<p class="tag">Beta is estimated once over the whole materialized path, never
inside the selected window: a descriptive window may slice what is displayed and
may not move a coefficient.</p>

<h2 id="controls">Selected controls</h2>
{_table("Controls set for this path", ("Control", "Value"), controls, empty="No controls.")}
<p class="tag">{html.escape(control_note)}</p>

<h2 id="schedule">Sleeve schedule guard</h2>
{schedule_section}

<h2 id="window">Study window guard</h2>
<div class="panel"><dl class="facts">
<dt>Requested window</dt><dd>{window.requested_start.isoformat()} to
{window.requested_end.isoformat()} {clamp_note}</dd>
<dt>Honoured window</dt><dd>{window.selected_start.isoformat()} to
{window.selected_end.isoformat()} ({window.selected_session_count} formations)</dd>
<dt>Full-support reference</dt><dd>{window.full_support_start.isoformat()} to
{window.full_support_end.isoformat()} ({window.full_support_session_count} formations)</dd>
<dt>Disposition</dt><dd class="warn">{html.escape(window.disposition)} &mdash; cannot select,
promote a default, or carry claim authority</dd>
<dt>Prefix retained</dt><dd>{window.prefix_formation_count} formations before the window,
never discarded</dd>
</dl></div>

<h2 id="coverage">Per-owner coverage and common watermark</h2>
{
        _table(
            "Exact support by owner",
            ("Owner", "Lane", "First", "Last", "Sessions"),
            coverage_rows,
            empty="No coverage resolved.",
        )
    }
<p class="tag">Common watermark {coverage.common_watermark_start.isoformat()} to
{coverage.common_watermark_end.isoformat()} ({coverage.common_session_count} sessions).
Alpha's frozen training window is {coverage.alpha_training_window_sessions} sessions and is
not a study bound.</p>

<h2 id="cost">Cost evidence ladder</h2>
{
        _table(
            "Fixed evidence ladder, per side and one-way turnover",
            ("Per side", "Platform one-way"),
            ladder,
            empty="No ladder.",
        )
    }


<h2 id="unit">{html.escape(spec.report_unit)} over the selected window</h2>
{
        _table(
            f"{spec.report_unit} rows",
            ("Period", "Return"),
            unit_rows,
            empty="No formations fall inside the selected window.",
        )
    }
{_sparkline(report.window_unit_rows)}

<h2 id="risk">Risk attribution at the final formation</h2>
{
        _table(
            "Industry exposure and variance contribution",
            ("Industry", "Exposure", "Variance contribution"),
            risk_rows,
            empty="No admitted Risk attribution for this path.",
        )
    }

<h2 id="identity">Identities</h2>
<dl class="facts">
<dt>Request</dt><dd><code>{spec.spec_hash}</code></dd>
<dt>Holdings configuration</dt><dd><code>{spec.holdings_spec_hash}</code></dd>
<dt>Program</dt><dd><code>{report.program_hash}</code></dd>
<dt>Execution ledger</dt><dd><code>{report.execution_ledger_hash}</code></dd>
<dt>Economic ledger</dt><dd><code>{report.economic_ledger_hash}</code></dd>
<dt>Benchmark comparison</dt><dd><code>{report.benchmark_comparison_hash}</code></dd>
<dt>Report</dt><dd><code>{report.report_hash}</code></dd>
<dt>Coverage</dt><dd><code>{coverage.coverage_hash}</code></dd>
</dl>

<h2 id="limitations">Limitations</h2>
<ul class="plain">{limitations}</ul>

<h2 id="refusals">Refused controls</h2>
<p class="tag">{refusals}</p>
</main>"""


__all__ = ["format_book_change", "format_book_weight", "render_portfolio_research_html"]
